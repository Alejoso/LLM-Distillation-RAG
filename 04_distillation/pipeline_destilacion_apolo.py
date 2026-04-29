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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    for m in models:
        if m is not None:
            del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    logging.info(f"[{experiment_name}] Teacher cargado. Generando datos (top_k={top_k})...")

    total = len(data)
    with open(distilled_path, "w", encoding="utf-8") as out:
        for idx, sample in enumerate(data):
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
        prompt_len = min(prompt_tokens["input_ids"].size(1), self.max_len)
        pad_id = self.tokenizer.pad_token_id
        for j in range(prompt_len, self.max_len):
            if input_ids[j].item() != pad_id:
                response_mask[j] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "teacher_logits": teacher_logits,
            "response_mask": response_mask,
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

    student = AutoModelForCausalLM.from_pretrained(
        config["student_name"], torch_dtype=torch.float16, device_map="auto"
    )

    student_vocab = student.config.vocab_size
    dataset = DistillationDataset(
        distilled_path, tokenizer, config["max_len"], student_vocab
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=config["lr"])
    start_epoch, global_step = 0, 0

    if ckpt_file.exists():
        logging.info(f"[{experiment_name}] Reanudando desde checkpoint...")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
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

            student_logits = student(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

            # Shift causal
            s_s = student_logits[:, :-1, :]
            s_t = teacher_logits[:, 1:, :]
            s_labels = input_ids[:, 1:]
            s_mask = response_mask[:, 1:]

            # Soft loss
            t_probs = F.softmax(s_t / T, dim=-1)
            s_log_probs = F.log_softmax(s_s / T, dim=-1)
            soft_tok = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=-1)
            soft_loss = (soft_tok * s_mask).sum() / (s_mask.sum() + 1e-8) * (T**2)

            # Hard loss
            hard_tok = F.cross_entropy(
                s_s.reshape(-1, s_s.size(-1)),
                s_labels.reshape(-1),
                reduction="none",
            ).view_as(s_labels)
            hard_loss = (hard_tok * s_mask).sum() / (s_mask.sum() + 1e-8)

            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"[{experiment_name}] NaN/Inf paso {global_step}, skip")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            valid_batches += 1
            last_soft = soft_loss.item()
            last_hard = hard_loss.item()

            if global_step % config["save_steps"] == 0:
                torch.save(
                    {"model": student.state_dict(), "optimizer": optimizer.state_dict(),
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
        logging.info(
            f"[{experiment_name}] Epoca {epoch+1}/{total_epochs} | "
            f"avg_loss={avg_loss:.4f} | soft={last_soft:.4f} | hard={last_hard:.4f}"
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
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    return extract_response(tokenizer.decode(out[0], skip_special_tokens=True))


def judge_response(
    judge_model, judge_tokenizer, device, question: str, answer: str,
    reference: str = "", max_new: int = 512,
) -> dict:
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


def evaluate_models(output_dir: Path, config: dict, experiments_run: list[str]):
    """Evalua modelo base + cada experimento destilado."""
    processed_path = output_dir / "processed.json"
    data = load_json(processed_path)
    device = get_device()

    # --- Definir modelos a evaluar ---
    models_to_eval = {}

    # A) Modelo base (student sin destilacion)
    models_to_eval["base_student"] = {
        "path": config["student_name"],  # desde HuggingFace directamente
        "label": "Student base (sin destilacion)",
    }

    # B) Modelos destilados
    for exp in experiments_run:
        final_dir = output_dir / exp / "final_model"
        if final_dir.exists():
            models_to_eval[exp] = {
                "path": str(final_dir),
                "label": f"Student destilado ({exp})",
            }

    # --- Cargar juez ---
    logging.info(f"Cargando modelo juez: {config['judge_name']}...")
    judge_tokenizer = AutoTokenizer.from_pretrained(config["judge_name"])
    if judge_tokenizer.pad_token is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token
    judge_model = AutoModelForCausalLM.from_pretrained(
        config["judge_name"], torch_dtype=torch.float16, device_map="auto"
    )
    judge_model.eval()
    logging.info("Juez cargado.")

    all_results = {}

    for model_key, model_info in models_to_eval.items():
        logging.info(f"Evaluando: {model_info['label']}...")

        # Cargar modelo a evaluar
        eval_tokenizer = AutoTokenizer.from_pretrained(model_info["path"])
        if eval_tokenizer.pad_token is None:
            eval_tokenizer.pad_token = eval_tokenizer.eos_token
        eval_model = AutoModelForCausalLM.from_pretrained(
            model_info["path"], torch_dtype=torch.float16, device_map="auto"
        )
        eval_model.eval()

        results = []
        for idx, sample in enumerate(data):
            instruction = sample["instruction"]
            reference = sample.get("reference", "")

            # Generar respuesta
            answer = generate_model_response(
                eval_model, eval_tokenizer, device, instruction
            )

            # Evaluar con juez
            scores = judge_response(
                judge_model, judge_tokenizer, device,
                instruction, answer, reference
            )

            weighted = compute_weighted_score(scores)
            results.append({
                "id": sample.get("id", idx),
                "instruction": instruction,
                "answer": answer,
                "reference": reference,
                "scores": scores,
                "weighted_score": weighted,
            })

            if (idx + 1) % 5 == 0 or (idx + 1) == len(data):
                logging.info(f"  [{model_key}] Evaluado: {idx+1}/{len(data)}")

        # Calcular promedios
        scored = [r for r in results if r["scores"]]
        avg_scores = {}
        if scored:
            for dim in JUDGE_WEIGHTS:
                key = f"{dim}_score"
                vals = [r["scores"].get(key, 0) for r in scored]
                avg_scores[f"avg_{dim}"] = round(sum(vals) / len(vals), 4)
            weighted_vals = [r["weighted_score"] for r in scored]
            avg_scores["avg_weighted"] = round(sum(weighted_vals) / len(weighted_vals), 4)

        summary = {
            "model_key": model_key,
            "label": model_info["label"],
            "num_evaluated": len(results),
            "num_scored": len(scored),
            "averages": avg_scores,
            "results": results,
        }
        all_results[model_key] = summary

        # Guardar resultados individuales
        save_json(summary, output_dir / "eval" / f"{model_key}_results.json")
        logging.info(
            f"  [{model_key}] Promedios: {avg_scores}"
        )

        free_model(eval_model)

    free_model(judge_model)

    # --- Tabla comparativa ---
    comparison = {
        "experiments": experiments_run,
        "models": {},
    }
    for mk, ms in all_results.items():
        comparison["models"][mk] = {
            "label": ms["label"],
            "averages": ms["averages"],
        }

    save_json(comparison, output_dir / "eval" / "comparison.json")
    save_report(output_dir, "04_evaluation", comparison)

    # Imprimir tabla
    logging.info("=" * 70)
    logging.info("RESULTADOS COMPARATIVOS")
    logging.info("=" * 70)
    header = f"{'Modelo':<35} {'Acc':>6} {'Rel':>6} {'Com':>6} {'Cla':>6} {'Total':>7}"
    logging.info(header)
    logging.info("-" * 70)
    for mk, ms in all_results.items():
        avgs = ms.get("averages", {})
        logging.info(
            f"{ms['label']:<35} "
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

    if step_done(output_dir, step):
        logging.info(f"[SKIP] {step}")
    else:
        logging.info(f"[RUN]  {step}")
        evaluate_models(output_dir, config, experiments_run)
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
    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
