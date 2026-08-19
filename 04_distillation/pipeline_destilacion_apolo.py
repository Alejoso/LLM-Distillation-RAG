#!/usr/bin/env python3
"""
Pipeline unificado de destilacion de conocimiento para el cluster Apolo (EAFIT).

Ejecuta dos experimentos de destilacion (con y sin RAG) y evalua comparativamente
tres modelos: base (sin destilacion), destilado sin RAG, destilado con RAG.

Uso:
    python pipeline_destilacion_apolo.py --data_dir ./data --output_dir ./outputs
    python pipeline_destilacion_apolo.py --data_dir ./data --output_dir ./outputs --reset
    python pipeline_destilacion_apolo.py --data_dir ./data --output_dir ./outputs --skip_rag
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
import gc

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Guardrail determinista pre-juez (answer_guardrails.py vive en judge_calibration/).
# Detecta respuestas degeneradas (vacio, eco, circular, abstencion) y les asigna
# un score fijo SIN llamar al LLM, evitando el artefacto "student > teacher" en el
# que un TinyLlama con salida truncada/degenerada recibia 5/5 del juez sesgado.
sys.path.insert(0, str(Path(__file__).resolve().parent / "judge_calibration"))
from answer_guardrails import screen_answer  # noqa: E402

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
DEFAULTS = {
    "seed": 42,
    "teacher_name": "meta-llama/Llama-2-7b-chat-hf",
    "student_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "epochs": 3,
    "batch_size": 2,
    "lr": 2e-5,
    "temperature": 4.0,
    "alpha": 0.7,
    "max_len": 2048,
    "max_new_tokens": 512,
    "top_k_logits": 50,
    "save_steps": 1000,
    "rag_top_k_initial": 10,
    "rag_top_k_final": 2,
    # Modelo juez para evaluacion (se usa el teacher por disponibilidad en Apolo)
    "judge_name": "meta-llama/Llama-2-7b-chat-hf",
    "judge_max_new_tokens": 512,
}

# Pesos del LLM-as-Judge (alineados con el articulo)
JUDGE_WEIGHTS = {
    "accuracy": 0.4,
    "relevance": 0.3,
    "completeness": 0.2,
    "clarity": 0.1,
}

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def checkpoint_path(output_dir: Path, step_name: str) -> Path:
    return output_dir / "checkpoints" / f"{step_name}.done"


def step_done(output_dir: Path, step_name: str) -> bool:
    return checkpoint_path(output_dir, step_name).exists()


def mark_done(output_dir: Path, step_name: str):
    cp = checkpoint_path(output_dir, step_name)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def clear_checkpoints(output_dir: Path):
    cp_dir = output_dir / "checkpoints"
    if cp_dir.exists():
        for f in cp_dir.glob("*.done"):
            f.unlink()
    logging.info("Checkpoints eliminados.")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(output_dir / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_report(output_dir: Path, name: str, metrics: dict):
    rdir = output_dir / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, rdir / f"{name}.json")
    logging.info(f"Reporte: {rdir / f'{name}.json'}")


def free_model(*models):
    """Libera modelos/tensores de GPU agresivamente.
    
    Mueve a CPU primero (rompe referencias de CUDA), borra,
    fuerza GC de Python, y limpia el caching allocator de PyTorch.
    """
    for m in models:
        if m is None:
            continue
        # Si es un nn.Module, mandarlo a CPU primero suelta los buffers GPU
        if hasattr(m, "cpu"):
            try:
                m.cpu()
            except Exception:
                pass
        del m
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ---------------------------------------------------------------------------
# HuggingFace Auth
# ---------------------------------------------------------------------------
def hf_login_if_needed():
    token = os.getenv("HF_TOKEN")
    if token:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        logging.info("HuggingFace login exitoso.")
    else:
        logging.info("HF_TOKEN no encontrado, usando credenciales en cache.")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def build_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"


def build_rag_prompt(instruction: str, contexts: list[str]) -> str:
    block = "\n\n---\n\n".join(c for c in contexts if c)
    return (
        f"### Instruction:\n{instruction.strip()}\n\n"
        f"### Retrieved Context:\n{block}\n\n"
        f"### Response:\n"
    )


def extract_response(text: str) -> str:
    if "### Response:" in text:
        return text.split("### Response:")[-1].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# RAG (opcional)
# ---------------------------------------------------------------------------
class RAGRetriever:
    def __init__(self, chroma_path: Path, top_k_initial: int, top_k_final: int):
        self.chroma_path = chroma_path
        self.top_k_initial = top_k_initial
        self.top_k_final = top_k_final
        self._db = None
        self._reranker = None

    def _init(self):
        if self._db is not None:
            return True
        try:
            from langchain_chroma import Chroma

            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_rag"))
            from create_db.define_BGEM3_embeddings import BgeM3Embeddings
            from query_db.reranker_BGE import Reranker

            self._db = Chroma(
                persist_directory=str(self.chroma_path),
                embedding_function=BgeM3Embeddings(),
            )
            self._reranker = Reranker()
            logging.info(f"RAG inicializado desde {self.chroma_path}")
            return True
        except Exception as e:
            logging.warning(f"No se pudo inicializar RAG: {e}")
            return False

    def retrieve(self, instruction: str) -> list[str]:
        if not instruction.strip():
            return []
        if not self._init():
            return []
        try:
            results = self._db.similarity_search_with_relevance_scores(
                instruction, k=self.top_k_initial
            )
            if not results:
                return []
            ranked = self._reranker.rerank_similarity_results(
                instruction, results, self.top_k_final
            )
            return [r["text"].strip() for r in ranked if r.get("text", "").strip()]
        except Exception as e:
            logging.warning(f"RAG retrieval fallo: {e}")
            return []


# ---------------------------------------------------------------------------
# Paso 1: Procesar dataset (compartido)
# ---------------------------------------------------------------------------
def process_dataset(data_dir: Path, output_dir: Path) -> list:
    raw_path = data_dir / "raw_train.json"
    processed_path = output_dir / "processed.json"

    if not raw_path.exists():
        raise FileNotFoundError(f"No se encontro {raw_path}")

    data = load_json(raw_path)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError("Dataset debe ser lista o dict con clave 'data'.")

    processed = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        instruction = (item.get("instruction") or "").strip()
        if not instruction:
            continue
        processed.append({
            "id": i,
            "instruction": instruction,
            "reference": (item.get("response") or "").strip(),
        })

    save_json(processed, processed_path)
    save_report(output_dir, "01_process_dataset", {
        "total_raw": len(data),
        "total_valid": len(processed),
    })
    logging.info(f"Dataset: {len(processed)}/{len(data)} muestras validas.")
    return processed


# ---------------------------------------------------------------------------
# Paso 2: Generar datos de destilacion (teacher inference)
# ---------------------------------------------------------------------------
def generate_distillation_data(
    output_dir: Path,
    experiment_name: str,
    config: dict,
    use_rag: bool,
    rag_retriever: RAGRetriever | None,
):
    processed_path = output_dir / "processed.json"
    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    distilled_path = exp_dir / "train.jsonl"

    data = load_json(processed_path)
    max_len = config["max_len"]
    max_new_tokens = config["max_new_tokens"]
    top_k = config["top_k_logits"]
    device = get_device()

    logits_dir = exp_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"[{experiment_name}] Cargando teacher: {config['teacher_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["teacher_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        config["teacher_name"], torch_dtype=torch.float16, device_map="auto"
    )
    teacher.eval()
    # Detectar progreso previo para reanudar si se cayo
    already_done = {int(f.stem) for f in logits_dir.glob("*.pt")}
    start_idx = len(already_done) if already_done else 0
    if start_idx > 0:
        # Verificar que el JSONL tiene las mismas lineas que los .pt ya guardados
        existing_lines = 0
        if distilled_path.exists():
            with open(distilled_path, "r", encoding="utf-8") as f:
                existing_lines = sum(1 for _ in f)
        if existing_lines != start_idx:
            # Reconstruir JSONL desde los datos ya procesados para mantener consistencia
            logging.warning(
                f"[{experiment_name}] JSONL tiene {existing_lines} lineas pero hay "
                f"{start_idx} logits .pt. Reconstruyendo JSONL..."
            )
            with open(distilled_path, "w", encoding="utf-8") as out:
                for i in range(start_idx):
                    sample = data[i]
                    instruction = sample["instruction"]
                    retrieved_contexts = []
                    if use_rag and rag_retriever:
                        retrieved_contexts = rag_retriever.retrieve(instruction)
                    prompt = (
                        build_rag_prompt(instruction, retrieved_contexts)
                        if retrieved_contexts
                        else build_prompt(instruction)
                    )
                    # Necesitamos regenerar teacher_output; no lo tenemos guardado
                    # Por seguridad, marcamos que debemos regenerar desde cero
                    pass
                # Si no podemos reconstruir el JSONL fiablemente, regenerar desde cero
                start_idx = 0
                already_done = set()
                # Limpiar .pt huerfanos
                for f in logits_dir.glob("*.pt"):
                    f.unlink()
            logging.warning(f"[{experiment_name}] Regenerando desde cero por inconsistencia.")

        else:
            logging.info(
                f"[{experiment_name}] Reanudando desde muestra {start_idx}/{len(data)} "
                f"({start_idx} ya procesadas)."
            )

    logging.info(f"[{experiment_name}] Teacher cargado. Generando datos (top_k={top_k})...")

    total = len(data)
    file_mode = "a" if start_idx > 0 else "w"
    with open(distilled_path, file_mode, encoding="utf-8") as out:
        for idx in range(start_idx, total):
            sample = data[idx]
            instruction = sample["instruction"]
            retrieved_contexts = []

            if use_rag and rag_retriever:
                retrieved_contexts = rag_retriever.retrieve(instruction)

            prompt = (
                build_rag_prompt(instruction, retrieved_contexts)
                if retrieved_contexts
                else build_prompt(instruction)
            )

            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=max_len
            ).to(device)

            with torch.no_grad():
                gen_ids = teacher.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            teacher_output = extract_response(full_text) or "[EMPTY_RESPONSE]"
            # NOTA: el EOS lo aprende el student via el eos_mask de
            # DistillationDataset (hard-CE sobre la 1a posicion pad/EOS tras la
            # respuesta). Funciona igual reusando estos datos/logits o
            # regenerandolos, sin depender de que el tokenizer parsee un "</s>"
            # de texto -> evita inyectar el string literal al entrenamiento.

            # Logits sobre prompt+respuesta completa
            full_tokens = tokenizer(
                prompt + teacher_output,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_len,
            ).to(device)

            with torch.no_grad():
                logits = teacher(**full_tokens).logits

            logits_clamped = torch.clamp(logits.squeeze(0), -10, 10)
            top_values, top_indices = logits_clamped.topk(top_k, dim=-1)
            torch.save({
                "values": top_values.half().cpu(),
                "indices": top_indices.cpu(),
            }, logits_dir / f"{idx}.pt")

            record = {
                "id": sample.get("id", idx),
                "instruction": instruction,
                "prompt": prompt,
                "used_rag": bool(retrieved_contexts),
                "retrieved_contexts": retrieved_contexts,
                "teacher_output": teacher_output,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            if (idx + 1) % 5 == 0 or (idx + 1) == total:
                logging.info(f"  [{experiment_name}] Teacher inference: {idx+1}/{total}")

    free_model(teacher)
    save_report(output_dir, f"02_distill_data_{experiment_name}", {
        "experiment": experiment_name,
        "total_samples": total,
        "rag_enabled": use_rag,
        "top_k_logits": top_k,
    })
    logging.info(f"[{experiment_name}] Datos de destilacion generados.")


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------
class DistillationDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_len: int, vocab_size: int):
        with open(path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        self.logits_dir = path.parent / "logits"
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        full_text = sample["prompt"] + sample["teacher_output"]

        tokens = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        prompt_tokens = self.tokenizer(
            sample["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # Cargar top-K logits desde archivo binario y reconstruir tensor completo
        logits_data = torch.load(self.logits_dir / f"{idx}.pt", weights_only=True)
        top_values = logits_data["values"].float()  # (seq_len, top_k)
        top_indices = logits_data["indices"]         # (seq_len, top_k)

        seq_len = top_values.size(0)
        # Rellenar con -10 (valor minimo de clamp) para tokens no top-K
        teacher_logits = torch.full(
            (self.max_len, self.vocab_size), -10.0, dtype=torch.float32
        )
        actual_len = min(seq_len, self.max_len)
        teacher_logits[:actual_len].scatter_(
            1, top_indices[:actual_len], top_values[:actual_len]
        )

        response_mask = torch.zeros(self.max_len, dtype=torch.float32)
        eos_mask = torch.zeros(self.max_len, dtype=torch.float32)
        prompt_len = min(prompt_tokens["input_ids"].size(1), self.max_len)
        pad_id = self.tokenizer.pad_token_id
        last_resp = prompt_len - 1
        for j in range(prompt_len, self.max_len):
            if input_ids[j].item() != pad_id:
                response_mask[j] = 1.0
                last_resp = j
        # Ensenar al student a TERMINAR: la primera posicion de pad tras la
        # respuesta contiene el token EOS (pad_token == eos_token). La marcamos
        # SOLO para la hard-loss (CE hacia EOS). La soft-loss (KD) la ignora
        # porque no hay logits del teacher en esa posicion. Sin esto el student
        # nunca recibe gradiente para emitir EOS -> nunca para -> degenera.
        eos_pos = last_resp + 1
        if prompt_len <= eos_pos < self.max_len and input_ids[eos_pos].item() == pad_id:
            eos_mask[eos_pos] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "teacher_logits": teacher_logits,
            "response_mask": response_mask,
            "eos_mask": eos_mask,
        }


# ---------------------------------------------------------------------------
# Paso 3: Entrenar student
# ---------------------------------------------------------------------------
def train_student(output_dir: Path, experiment_name: str, config: dict):
    exp_dir = output_dir / experiment_name
    distilled_path = exp_dir / "train.jsonl"
    ckpt_file = exp_dir / "training_last.pt"
    log_file = exp_dir / "training_logs.jsonl"
    device = get_device()

    logging.info(f"[{experiment_name}] Cargando student: {config['student_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["student_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pesos master en FP32 (V100 no tiene bf16 nativo); el forward usa FP16
    # via autocast + GradScaler. Entrenar con torch_dtype=float16 directo
    # corrompe los pesos al primer optimizer.step() y produce NaN en todo
    # el grafo desde el segundo batch en adelante.
    student = AutoModelForCausalLM.from_pretrained(
        config["student_name"], torch_dtype=torch.float32, device_map="auto"
    )

    student_vocab = student.config.vocab_size
    dataset = DistillationDataset(
        distilled_path, tokenizer, config["max_len"], student_vocab
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=config["lr"])
    scaler = torch.cuda.amp.GradScaler()
    start_epoch, global_step = 0, 0

    if ckpt_file.exists():
        logging.info(f"[{experiment_name}] Reanudando desde checkpoint...")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["step"]

    student.train()
    alpha = config["alpha"]
    T = config["temperature"]
    total_epochs = config["epochs"]

    logging.info(
        f"[{experiment_name}] Entrenamiento: {total_epochs} epocas, "
        f"bs={config['batch_size']}, lr={config['lr']}, alpha={alpha}, T={T}"
    )

    avg_loss = 0.0
    for epoch in range(start_epoch, total_epochs):
        epoch_loss, valid_batches = 0.0, 0
        last_soft, last_hard = None, None

        for batch in loader:
            global_step += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            teacher_logits = batch["teacher_logits"].to(device)
            response_mask = batch["response_mask"].to(device)
            eos_mask = batch["eos_mask"].to(device)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                student_logits = student(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits

                # Shift causal
                s_s = student_logits[:, :-1, :]
                s_t = teacher_logits[:, 1:, :]
                s_labels = input_ids[:, 1:]
                s_mask = response_mask[:, 1:]
                # Hard-loss tambien sobre la posicion EOS (KD no, no hay teacher ahi).
                h_mask = (response_mask + eos_mask)[:, 1:].clamp(max=1.0)

                # Soft/hard loss computados en FP32 fuera del autocast para
                # evitar overflow en exp() del softmax con vocab grande.
            s_s_f32 = s_s.float()
            t_probs = F.softmax(s_t / T, dim=-1)
            s_log_probs = F.log_softmax(s_s_f32 / T, dim=-1)
            soft_tok = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=-1)
            soft_loss = (soft_tok * s_mask).sum() / (s_mask.sum() + 1e-8) * (T**2)

            hard_tok = F.cross_entropy(
                s_s_f32.reshape(-1, s_s_f32.size(-1)),
                s_labels.reshape(-1),
                reduction="none",
            ).view_as(s_labels)
            hard_loss = (hard_tok * h_mask).sum() / (h_mask.sum() + 1e-8)

            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"[{experiment_name}] NaN/Inf paso {global_step}, skip")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            valid_batches += 1
            last_soft = soft_loss.item()
            last_hard = hard_loss.item()

            if global_step % config["save_steps"] == 0:
                torch.save(
                    {"model": student.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict(),
                     "epoch": epoch, "step": global_step},
                    ckpt_file,
                )

            with log_file.open("a", encoding="utf-8") as lf:
                lf.write(json.dumps({
                    "epoch": epoch, "step": global_step,
                    "loss": loss.item(), "soft": soft_loss.item(),
                    "hard": hard_loss.item(), "time": time.time(),
                }, ensure_ascii=False) + "\n")

        avg_loss = epoch_loss / max(valid_batches, 1)
        soft_str = f"{last_soft:.4f}" if last_soft is not None else "N/A"
        hard_str = f"{last_hard:.4f}" if last_hard is not None else "N/A"
        logging.info(
            f"[{experiment_name}] Epoca {epoch+1}/{total_epochs} | "
            f"avg_loss={avg_loss:.4f} | soft={soft_str} | hard={hard_str} | "
            f"valid_batches={valid_batches}"
        )
        if valid_batches == 0:
            raise RuntimeError(
                f"[{experiment_name}] Epoca {epoch+1}: 0 batches validos "
                f"(todos NaN/Inf). Entrenamiento corrupto, abortando."
            )

    # Guardar modelo final
    final_dir = exp_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    save_report(output_dir, f"03_train_{experiment_name}", {
        "experiment": experiment_name,
        "epochs": total_epochs,
        "total_steps": global_step,
        "final_avg_loss": avg_loss,
    })
    logging.info(f"[{experiment_name}] Modelo guardado en {final_dir}")

    del optimizer, scaler, loader, dataset
    free_model(student)

# ---------------------------------------------------------------------------
# Paso 4: Evaluacion comparativa (LLM-as-Judge)
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """You are a strict evaluator. Given a question and an answer, score the answer on four dimensions.
Return ONLY valid JSON with these exact keys:
- accuracy_score (1-5): 1=completely incorrect, 5=fully correct
- relevance_score (1-5): 1=off-topic, 5=directly answers all parts
- completeness_score (1-5): 1=very incomplete, 5=fully complete
- clarity_score (1-5): 1=very unclear, 5=very clear
- one_sentence_summary: exactly one sentence summary

Output ONLY valid JSON. No markdown. No extra text."""


def generate_model_response(model, tokenizer, device, instruction: str, max_new: int = 256) -> str:
    prompt = build_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,              # greedy: eval determinista y reproducible
            repetition_penalty=1.3,       # corta los loops degenerados
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Decodificar solo los tokens NUEVOS (no el prompt) y cortar en EOS.
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def judge_response(
    judge_model, judge_tokenizer, device, question: str, answer: str,
    reference: str = "", max_new: int = 512, use_guardrail: bool = True,
) -> dict:
    # Guardrail determinista: si el candidato es degenerado (vacio, eco de la
    # pregunta, circular, o abstencion) se asigna el score fijo sin consultar al
    # juez LLM. Es de alta precision: no dispara sobre respuestas que contienen el
    # hecho de la referencia. Devuelve el marcador _guard para auditoria.
    if use_guardrail:
        guard = screen_answer(question, answer, reference)
        if guard is not None:
            return guard

    user_content = f"Question:\n{question}\n"
    if reference:
        user_content += f"\nReference answer:\n{reference}\n"
    user_content += f"\nAnswer to evaluate:\n{answer}"

    # Formato chat para Llama-2
    prompt = (
        f"<s>[INST] <<SYS>>\n{JUDGE_SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_content} [/INST]"
    )

    inputs = judge_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        out = judge_model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=judge_tokenizer.pad_token_id,
        )

    raw = judge_tokenizer.decode(out[0], skip_special_tokens=True)
    # Extraer JSON de la respuesta
    if "[/INST]" in raw:
        raw = raw.split("[/INST]")[-1].strip()

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except (json.JSONDecodeError, AttributeError):
        pass

    logging.warning(f"Judge no retorno JSON valido. Raw: {raw[:200]}")
    return {}


def compute_weighted_score(scores: dict) -> float:
    total = 0.0
    for dim, weight in JUDGE_WEIGHTS.items():
        total += weight * scores.get(f"{dim}_score", 0)
    return round(total, 4)


def _load_eval_subset(eval_dir: Path, data: list, subset_size: int | None, seed: int) -> list:
    """Devuelve un subset aleatorio determinista compartido entre todas las fases.

    Si ``subset_size`` es None, <=0 o >= len(data), se usa el conjunto completo.
    El subset se persiste a ``eval_subset.json`` para que reanudaciones y otras
    fases evaluen exactamente los mismos items. Borrar ese archivo (o el
    directorio eval/ entero) regenera el subset con el seed actual.
    """
    if subset_size is None or subset_size <= 0 or subset_size >= len(data):
        logging.info(f"Eval set completo: {len(data)} items.")
        return data

    subset_path = eval_dir / "eval_subset.json"
    if subset_path.exists():
        subset = load_json(subset_path)
        logging.info(
            f"Eval subset reusado: {len(subset)} items desde {subset_path.name} "
            f"(borra eval/ para regenerar)."
        )
        return subset

    rng = random.Random(seed)
    subset = rng.sample(data, subset_size)
    save_json(subset, subset_path)
    logging.info(
        f"Eval subset creado: {len(subset)} items (seed={seed}) -> {subset_path.name}"
    )
    return subset


def _read_done_items(items_path: Path) -> tuple[set, list]:
    """Lee items ya evaluados (JSONL) y devuelve (set de ids, lista de records)."""
    if not items_path.exists():
        return set(), []
    records = []
    with items_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    done_ids = {r["id"] for r in records if "id" in r}
    return done_ids, records


def _summarize(records: list, model_key: str, label: str) -> dict:
    scored = [r for r in records if r.get("scores")]
    avg_scores = {}
    if scored:
        for dim in JUDGE_WEIGHTS:
            key = f"{dim}_score"
            vals = [r["scores"].get(key, 0) for r in scored]
            avg_scores[f"avg_{dim}"] = round(sum(vals) / len(vals), 4)
        weighted_vals = [r.get("weighted_score", 0) for r in scored]
        avg_scores["avg_weighted"] = round(sum(weighted_vals) / len(weighted_vals), 4)
    return {
        "model_key": model_key,
        "label": label,
        "num_evaluated": len(records),
        "num_scored": len(scored),
        "averages": avg_scores,
        "results": records,
    }


def _rebuild_comparison(eval_dir: Path, experiments_run: list[str]) -> dict:
    """Reconstruye comparison.json a partir de los *_results.json que existan."""
    comparison = {"experiments": experiments_run, "models": {}}
    for path in sorted(eval_dir.glob("*_results.json")):
        try:
            summary = load_json(path)
        except Exception:
            continue
        mk = summary.get("model_key") or path.stem.replace("_results", "")
        comparison["models"][mk] = {
            "label": summary.get("label", mk),
            "averages": summary.get("averages", {}),
            "num_evaluated": summary.get("num_evaluated", 0),
        }
    save_json(comparison, eval_dir / "comparison.json")
    return comparison


def evaluate_models(
    output_dir: Path,
    config: dict,
    experiments_run: list[str],
    only_models: list[str] | None = None,
    eval_subset_size: int | None = None,
    eval_seed: int = 42,
):
    """Evalua modelo base + cada experimento destilado.

    Subset y reanudacion:
    - ``eval_subset_size``: si se proporciona, se evalua sobre un subset aleatorio
      determinista compartido por todas las fases (ver ``_load_eval_subset``).
    - Cada item se persiste a ``eval/{model_key}_items.jsonl`` al instante, de
      forma que un kill (timeout SLURM) pierde a lo sumo el item en curso.
    - ``only_models``: lista de model_keys (e.g. ['base_student', 'no_rag']) para
      correr solo esas fases. Permite lanzar 3 jobs SLURM en paralelo, uno por
      modelo. La comparacion se reconstruye desde los resultados disponibles.
    """
    processed_path = output_dir / "processed.json"
    data = load_json(processed_path)
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    data = _load_eval_subset(eval_dir, data, eval_subset_size, eval_seed)
    total_subset = len(data)

    # --- Definir modelos a evaluar ---
    models_to_eval = {
        "base_student": {
            "path": config["student_name"],
            "label": "Student base (sin destilacion)",
        },
    }
    for exp in experiments_run:
        final_dir = output_dir / exp / "final_model"
        if final_dir.exists():
            models_to_eval[exp] = {
                "path": str(final_dir),
                "label": f"Student destilado ({exp})",
            }

    if only_models:
        keep = set(only_models)
        unknown = keep - set(models_to_eval.keys())
        if unknown:
            logging.warning(f"only_models ignorados (desconocidos): {sorted(unknown)}")
        models_to_eval = {k: v for k, v in models_to_eval.items() if k in keep}
        if not models_to_eval:
            logging.error("Ningun modelo a evaluar tras filtrar por only_models.")
            return _rebuild_comparison(eval_dir, experiments_run)

    # --- Determinar trabajo pendiente por modelo ---
    pending = {}
    for mk in models_to_eval:
        items_path = eval_dir / f"{mk}_items.jsonl"
        done_ids, _ = _read_done_items(items_path)
        pending[mk] = [s for s in data if s.get("id") not in done_ids]

    work_needed = any(remaining for remaining in pending.values())

    judge_model = None
    judge_tokenizer = None
    if work_needed:
        logging.info(f"Cargando modelo juez: {config['judge_name']}...")
        judge_tokenizer = AutoTokenizer.from_pretrained(config["judge_name"])
        if judge_tokenizer.pad_token is None:
            judge_tokenizer.pad_token = judge_tokenizer.eos_token
        judge_model = AutoModelForCausalLM.from_pretrained(
            config["judge_name"], torch_dtype=torch.float16, device_map="auto"
        )
        judge_model.eval()
        logging.info("Juez cargado.")

    for model_key, model_info in models_to_eval.items():
        items_path = eval_dir / f"{model_key}_items.jsonl"
        done_ids, prior_records = _read_done_items(items_path)
        remaining = pending[model_key]

        if not remaining:
            logging.info(
                f"  [{model_key}] Ya completo: {len(prior_records)}/{total_subset}."
            )
            summary = _summarize(prior_records, model_key, model_info["label"])
            save_json(summary, eval_dir / f"{model_key}_results.json")
            _rebuild_comparison(eval_dir, experiments_run)
            continue

        if done_ids:
            logging.info(
                f"  [{model_key}] Reanudando: {len(done_ids)} hechos, "
                f"faltan {len(remaining)}/{total_subset}."
            )
        else:
            logging.info(
                f"  [{model_key}] Empezando evaluacion sobre {total_subset} items."
            )

        logging.info(f"Evaluando: {model_info['label']}...")
        eval_tokenizer = AutoTokenizer.from_pretrained(model_info["path"])
        if eval_tokenizer.pad_token is None:
            eval_tokenizer.pad_token = eval_tokenizer.eos_token
        eval_model = AutoModelForCausalLM.from_pretrained(
            model_info["path"], torch_dtype=torch.float16, device_map="auto"
        )
        eval_model.eval()

        done_count = len(done_ids)
        with items_path.open("a", encoding="utf-8") as items_f:
            for sample in remaining:
                instruction = sample["instruction"]
                reference = sample.get("reference", "")

                answer = generate_model_response(
                    eval_model, eval_tokenizer, device, instruction
                )
                scores = judge_response(
                    judge_model, judge_tokenizer, device,
                    instruction, answer, reference
                )
                weighted = compute_weighted_score(scores)
                record = {
                    "id": sample.get("id"),
                    "instruction": instruction,
                    "answer": answer,
                    "reference": reference,
                    "scores": scores,
                    "weighted_score": weighted,
                }
                items_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                items_f.flush()
                done_count += 1

                if done_count % 5 == 0 or done_count == total_subset:
                    logging.info(f"  [{model_key}] Evaluado: {done_count}/{total_subset}")

        _, all_records = _read_done_items(items_path)
        summary = _summarize(all_records, model_key, model_info["label"])
        save_json(summary, eval_dir / f"{model_key}_results.json")
        logging.info(f"  [{model_key}] Promedios: {summary['averages']}")

        free_model(eval_model)
        _rebuild_comparison(eval_dir, experiments_run)

    if judge_model is not None:
        free_model(judge_model)

    comparison = _rebuild_comparison(eval_dir, experiments_run)
    save_report(output_dir, "04_evaluation", comparison)

    logging.info("=" * 70)
    logging.info("RESULTADOS COMPARATIVOS")
    logging.info("=" * 70)
    header = f"{'Modelo':<35} {'Acc':>6} {'Rel':>6} {'Com':>6} {'Cla':>6} {'Total':>7}"
    logging.info(header)
    logging.info("-" * 70)
    for mk, ms in comparison.get("models", {}).items():
        avgs = ms.get("averages", {})
        logging.info(
            f"{ms.get('label', mk):<35} "
            f"{avgs.get('avg_accuracy', 0):>6.2f} "
            f"{avgs.get('avg_relevance', 0):>6.2f} "
            f"{avgs.get('avg_completeness', 0):>6.2f} "
            f"{avgs.get('avg_clarity', 0):>6.2f} "
            f"{avgs.get('avg_weighted', 0):>7.4f}"
        )
    logging.info("=" * 70)

    return comparison


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------
def run_pipeline(args):
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)
    logging.info("=" * 60)
    logging.info("PIPELINE DE DESTILACION COMPARATIVA - APOLO (EAFIT)")
    logging.info("=" * 60)
    logging.info(f"data_dir:   {data_dir}")
    logging.info(f"output_dir: {output_dir}")
    logging.info(f"skip_rag:   {args.skip_rag}")
    logging.info(f"reset:      {args.reset}")

    config = {**DEFAULTS}

    if args.reset:
        clear_checkpoints(output_dir)

    set_seed(config["seed"])
    hf_login_if_needed()

    # Verificar disponibilidad de RAG
    rag_retriever = None
    rag_available = False
    if not args.skip_rag:
        chroma_path = data_dir.parent / "chroma"
        if chroma_path.exists():
            rag_retriever = RAGRetriever(
                chroma_path, config["rag_top_k_initial"], config["rag_top_k_final"]
            )
            rag_available = True
        else:
            logging.warning(f"ChromaDB no encontrada en {chroma_path}. Experimento con RAG no se ejecutara.")

    if args.only_eval:
        logging.info("[ONLY EVAL] Saltando fases 01-03 (distillation + training).")
    else:
        # ===== PASO 1: Procesar dataset (compartido) =====
        step = "01_process_dataset"
        if step_done(output_dir, step):
            logging.info(f"[SKIP] {step}")
        else:
            logging.info(f"[RUN]  {step}")
            process_dataset(data_dir, output_dir)
            mark_done(output_dir, step)

        # ===== PASO 2A: Generar datos de destilacion SIN RAG =====
        step = "02_distill_no_rag"
        if step_done(output_dir, step):
            logging.info(f"[SKIP] {step}")
        else:
            logging.info(f"[RUN]  {step}")
            generate_distillation_data(output_dir, "no_rag", config, False, None)
            mark_done(output_dir, step)

        # ===== PASO 2B: Generar datos de destilacion CON RAG =====
        if rag_available:
            step = "02_distill_with_rag"
            if step_done(output_dir, step):
                logging.info(f"[SKIP] {step}")
            else:
                logging.info(f"[RUN]  {step}")
                generate_distillation_data(output_dir, "with_rag", config, True, rag_retriever)
                mark_done(output_dir, step)

        # ===== PASO 3A: Entrenar student SIN RAG =====
        step = "03_train_no_rag"
        if step_done(output_dir, step):
            logging.info(f"[SKIP] {step}")
        else:
            logging.info(f"[RUN]  {step}")
            train_student(output_dir, "no_rag", config)
            mark_done(output_dir, step)

        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            logging.info(
                f"[MEM] GPU tras no_rag: "
                f"{free_b/1e9:.2f}/{total_b/1e9:.2f} GB libres"
            )

        # ===== PASO 3B: Entrenar student CON RAG =====
        if rag_available:
            step = "03_train_with_rag"
            if step_done(output_dir, step):
                logging.info(f"[SKIP] {step}")
            else:
                logging.info(f"[RUN]  {step}")
                train_student(output_dir, "with_rag", config)
                mark_done(output_dir, step)

    # ===== PASO 4: Evaluacion comparativa =====
    step = "04_evaluation"
    experiments_run = ["no_rag"]
    if rag_available:
        experiments_run.append("with_rag")

    only_models = None
    if args.only_models:
        only_models = [m.strip() for m in args.only_models.split(",") if m.strip()]

    if step_done(output_dir, step) and not args.only_eval and not only_models:
        logging.info(f"[SKIP] {step}")
    else:
        logging.info(f"[RUN]  {step}")
        evaluate_models(
            output_dir, config, experiments_run,
            only_models=only_models,
            eval_subset_size=args.eval_subset_size,
            eval_seed=args.eval_seed,
        )
        # Solo marcamos como done si se evaluaron TODOS los modelos esperados.
        # Con --only_models / --only_eval por fase, dejamos el checkpoint abierto
        # para permitir corridas adicionales.
        if not only_models:
            mark_done(output_dir, step)

    logging.info("=" * 60)
    logging.info("PIPELINE COMPLETADO")
    logging.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline de destilacion comparativa para Apolo (EAFIT)"
    )
    p.add_argument("--data_dir", default="./data",
                    help="Directorio con raw_train.json")
    p.add_argument("--output_dir", default="./outputs",
                    help="Directorio de salida")
    p.add_argument("--reset", action="store_true",
                    help="Ignorar checkpoints, ejecutar desde cero")
    p.add_argument("--skip_rag", action="store_true",
                    help="Saltar experimento con RAG (solo ejecutar sin RAG)")
    p.add_argument("--only_eval", action="store_true",
                    help="Saltar fases 01-03 y correr solo la evaluacion. "
                         "Asume que el entrenamiento ya termino.")
    p.add_argument("--only_models", default=None,
                    help="Lista coma-separada de model_keys a evaluar "
                         "(e.g. 'base_student' o 'no_rag,with_rag'). "
                         "Permite paralelizar la evaluacion en varios jobs SLURM.")
    p.add_argument("--eval_subset_size", type=int, default=1500,
                    help="Tamano del subset aleatorio para evaluacion. "
                         "Usar 0 para evaluar el dataset completo (NO recomendado: "
                         "63k items > 10 dias en accel-2). Default: 1500.")
    p.add_argument("--eval_seed", type=int, default=42,
                    help="Seed para el muestreo del subset de evaluacion.")
    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
