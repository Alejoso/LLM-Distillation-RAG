"""
Reproduccion local del bug de degeneracion del student (EOS nunca aprendido).

Aisla la causa raiz: el response_mask del pipeline excluye la posicion EOS de la
perdida (pad_token == eos_token), asi que el student NUNCA recibe gradiente para
terminar -> nunca para -> degenera (loops / ingles).

Compara dos variantes entrenando el MISMO subconjunto de parametros sobre unos
pocos pares (pregunta ES -> respuesta corta ES) reales de raw_train.json:
  - BUGGY: response_mask excluye EOS  (comportamiento actual del pipeline)
  - FIXED: response_mask + eos_mask   (el fix: hard-CE sobre la posicion EOS)

Nota honesta: es una micro-destilacion (CE, sin logits de teacher) sobre el
TinyLlama base para demostrar el MECANISMO, no la corrida completa de 76k en
Apolo. Lo decisivo aqui es: con EOS en la perdida el modelo aprende a PARAR.
"""
import json
import sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# raw_train.json vive en 04_distillation/data/ (un nivel arriba de este repro/).
RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "raw_train.json")
MAXLEN = 48
STEPS = 150
LR = 1e-4

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token  # igual que el pipeline
EOS = tok.eos_token_id


def build_prompt(instr):
    return f"### Instruction:\n{instr.strip()}\n\n### Response:\n"


# --- datos: pares reales cortos en espanol -------------------------------------
data = json.load(open(RAW))
data = data["data"] if isinstance(data, dict) else data
pairs = []
for d in data:
    r = (d.get("response") or "").strip()
    if 8 < len(r) < 45 and any(c in r.lower() for c in "áéíóúñ ") and "\n" not in r:
        pairs.append((d["instruction"], r))
    if len(pairs) == 8:
        break
print(f"\n{len(pairs)} pares de entrenamiento (ES, respuesta corta):")
for q, a in pairs:
    print(f"  Q: {q[:60]:<60} -> A: {a}")


def make_batch(train_eos):
    ids_all, mask_all = [], []
    for instr, resp in pairs:
        prompt = build_prompt(instr)
        ptoks = tok(prompt, truncation=True, max_length=MAXLEN)["input_ids"]
        ftoks = tok(prompt + resp, truncation=True, max_length=MAXLEN)["input_ids"]
        ids = ftoks + [tok.pad_token_id] * (MAXLEN - len(ftoks))
        ids = ids[:MAXLEN]
        m = [0.0] * MAXLEN
        plen = min(len(ptoks), MAXLEN)
        last = plen - 1
        for j in range(plen, MAXLEN):
            if ids[j] != tok.pad_token_id:
                m[j] = 1.0
                last = j
        if train_eos:  # marca la primera posicion EOS/pad tras la respuesta
            epos = last + 1
            if plen <= epos < MAXLEN and ids[epos] == tok.pad_token_id:
                m[epos] = 1.0
        ids_all.append(ids)
        mask_all.append(m)
    return torch.tensor(ids_all), torch.tensor(mask_all)


def train_variant(train_eos):
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(device)
    # Para que sea rapido y ligero: entrena solo embeddings/lm_head (atadas) +
    # ultima capa. El punto es la MASCARA, identica en ambas variantes salvo EOS.
    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    model.get_input_embeddings().weight.requires_grad = True
    trainable.append(model.get_input_embeddings().weight)
    for p in model.model.layers[-1].parameters():
        p.requires_grad = True
        trainable.append(p)
    opt = torch.optim.AdamW([p for p in trainable if p.requires_grad], lr=LR)

    ids, masks = make_batch(train_eos)
    ids, masks = ids.to(device), masks.to(device)
    attn = (ids != tok.pad_token_id).long()
    model.train()
    for step in range(STEPS):
        logits = model(input_ids=ids, attention_mask=attn).logits
        sl = logits[:, :-1, :]
        lab = ids[:, 1:]
        mk = masks[:, 1:]
        ce = F.cross_entropy(
            sl.reshape(-1, sl.size(-1)), lab.reshape(-1), reduction="none"
        ).view_as(lab)
        loss = (ce * mk).sum() / (mk.sum() + 1e-8)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 30 == 0 or step == STEPS - 1:
            print(f"    step {step:3d} loss {loss.item():.3f}")
    model.eval()
    return model


def generate(model, instr, max_new=100, eval_decoding=False):
    # eval_decoding=False: greedy CRUDO -> expone si el modelo PARA por si mismo
    #   (efecto del fix de ENTRENAMIENTO).
    # eval_decoding=True: la config real de eval del pipeline (repetition_penalty
    #   + no_repeat_ngram_size) -> segunda capa que corta cualquier loop residual.
    enc = tok(build_prompt(instr), return_tensors="pt").to(device)
    extra = dict(repetition_penalty=1.3, no_repeat_ngram_size=3) if eval_decoding else {}
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new, do_sample=False,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
            **extra,
        )
    gen = out[0][enc["input_ids"].shape[1]:].tolist()
    stopped = EOS in gen
    n = gen.index(EOS) + 1 if stopped else len(gen)
    text = tok.decode(gen, skip_special_tokens=True).strip()
    return text, stopped, n


heldout = "¿Qué entidad expide la ley según el artículo 1?"

for name, train_eos in [("BUGGY (mascara actual, EOS excluido)", False),
                        ("FIXED (EOS entrenado via hard-CE)", True)]:
    print("\n" + "=" * 78)
    print(f"VARIANTE: {name}")
    print("=" * 78)
    model = train_variant(train_eos)
    print("\n  -- generacion greedy CRUDA (sin repetition_penalty) --")
    probes = [pairs[0][0], pairs[3][0], heldout]
    for q in probes:
        text, stopped, n = generate(model, q)
        flag = "PARA en EOS" if stopped else "NO PARA (llega al tope)"
        print(f"\n  Q: {q[:70]}")
        print(f"  [{flag}, {n} toks] {text[:220]}")
    if train_eos:
        print("\n  -- MISMO modelo FIXED + config de decoding de eval (2a capa) --")
        for q in probes:
            text, stopped, n = generate(model, q, eval_decoding=True)
            flag = "PARA en EOS" if stopped else "NO PARA (llega al tope)"
            print(f"\n  Q: {q[:70]}")
            print(f"  [{flag}, {n} toks] {text[:220]}")
    del model
    if device == "mps":
        torch.mps.empty_cache()

print("\n\nRESUMEN: si BUGGY 'NO PARA' y FIXED 'PARA en EOS', el fix de EOS es la cura.")
