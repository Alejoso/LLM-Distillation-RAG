#!/usr/bin/env python3
"""Evaluacion del teacher (con y sin RAG) usando el LLM-as-Judge.

El pipeline guarda en outputs/{no_rag,with_rag}/train.jsonl el ``teacher_output``
generado en la fase 02 (greedy, do_sample=False), junto con el ``id`` original.
Este script:

  1. Carga ``outputs/eval/eval_subset.json`` (los 1500 ids muestreados).
  2. Filtra los ``teacher_output`` por esos ids.
  3. Llama al juez sobre cada (question, teacher_output, reference).
  4. Guarda outputs/eval/teacher_{exp}_items.jsonl y teacher_{exp}_results.json.

Por defecto usa el prompt actual del juez (v1) para que los numeros sean
directamente comparables con base_student/no_rag/with_rag. Pasar
``--judge_prompt_version v2`` para usar el prompt mejorado del set de
calibracion.

Uso:
    python eval_teacher.py --output_dir ./outputs --experiments no_rag with_rag

Reanudacion: si teacher_{exp}_items.jsonl ya existe, se saltean los ids
ya evaluados (mismo comportamiento que pipeline_destilacion_apolo.evaluate_models).
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

# Reusar prompts del set de calibracion
sys.path.insert(0, str(Path(__file__).resolve().parent / "judge_calibration"))
from judge_prompts import build_judge_prompt  # noqa: E402

# torch/transformers se importan dentro de main() para permitir tests sin GPU.

JUDGE_WEIGHTS = {"accuracy": 0.4, "relevance": 0.3, "completeness": 0.2, "clarity": 0.1}
DIMENSIONS = list(JUDGE_WEIGHTS.keys())


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_done(items_path: Path) -> tuple[set, list]:
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
    return {r["id"] for r in records if "id" in r}, records


def judge(model, tokenizer, device, prompt: str, max_new: int) -> dict:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    if "[/INST]" in raw:
        raw = raw.split("[/INST]")[-1].strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        logging.warning(f"Juez sin JSON. Raw: {raw[:160]}")
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        logging.warning(f"JSON invalido del juez. Raw: {match.group(0)[:160]}")
        return {}


def weighted_score(scores: dict) -> float:
    return round(sum(w * scores.get(f"{d}_score", 0) for d, w in JUDGE_WEIGHTS.items()), 4)


def summarize(records: list, model_key: str, label: str) -> dict:
    scored = [r for r in records if r.get("scores")]
    averages = {}
    if scored:
        for dim in DIMENSIONS:
            key = f"{dim}_score"
            vals = [r["scores"].get(key, 0) for r in scored]
            averages[f"avg_{dim}"] = round(sum(vals) / len(vals), 4)
        weighted_vals = [r.get("weighted_score", 0) for r in scored]
        averages["avg_weighted"] = round(sum(weighted_vals) / len(weighted_vals), 4)
    return {
        "model_key": model_key,
        "label": label,
        "num_evaluated": len(records),
        "num_scored": len(scored),
        "averages": averages,
        "results": records,
    }


def rebuild_comparison(eval_dir: Path):
    """Reconstruye comparison.json mezclando todos los *_results.json del directorio."""
    comparison = {"models": {}}
    experiments = []
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
        if mk in ("no_rag", "with_rag", "teacher_no_rag", "teacher_with_rag"):
            experiments.append(mk)
    comparison["experiments"] = sorted(set(experiments))
    save_json(comparison, eval_dir / "comparison.json")
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./outputs",
                        help="Directorio raiz del pipeline (con processed.json, eval/, no_rag/, with_rag/)")
    parser.add_argument("--experiments", nargs="+", default=["no_rag", "with_rag"],
                        choices=["no_rag", "with_rag"])
    parser.add_argument("--judge_name", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--judge_prompt_version", default="v1", choices=["v1", "v2"],
                        help="v1=prompt actual del pipeline; v2=prompt mejorado con rubric+few-shot")
    parser.add_argument("--judge_max_new_tokens", type=int, default=512)
    parser.add_argument("--eval_subset_size", type=int, default=1500,
                        help="Solo se usa si no existe eval/eval_subset.json. "
                             "Si existe, se reusa el subset para mantener comparabilidad.")
    parser.add_argument("--eval_seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "eval_teacher.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("=" * 60)
    logging.info("EVAL TEACHER (LLM-as-Judge)")
    logging.info(f"judge:         {args.judge_name}")
    logging.info(f"prompt:        {args.judge_prompt_version}")
    logging.info(f"experiments:   {args.experiments}")
    logging.info("=" * 60)

    # --- Determinar el subset ---
    subset_path = eval_dir / "eval_subset.json"
    if subset_path.exists():
        subset = load_json(subset_path)
        logging.info(f"Reusando eval_subset.json: {len(subset)} items.")
    else:
        # Cae al processed.json y muestrea con el mismo seed que el pipeline
        import random
        processed = load_json(output_dir / "processed.json")
        if args.eval_subset_size <= 0 or args.eval_subset_size >= len(processed):
            subset = processed
        else:
            rng = random.Random(args.eval_seed)
            subset = rng.sample(processed, args.eval_subset_size)
            save_json(subset, subset_path)
        logging.info(f"eval_subset creado: {len(subset)} items.")

    subset_by_id = {item["id"]: item for item in subset}

    # --- Cargar juez ---
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logging.info(f"Cargando juez {args.judge_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    logging.info("Juez cargado.")

    # --- Evaluar cada experimento ---
    for exp in args.experiments:
        model_key = f"teacher_{exp}"
        label = f"Teacher ({exp})"
        train_path = output_dir / exp / "train.jsonl"
        items_path = eval_dir / f"{model_key}_items.jsonl"

        if not train_path.exists():
            logging.warning(f"No existe {train_path}, salteando experimento {exp}.")
            continue

        # Cargar teacher_output desde train.jsonl, filtrar por subset
        teacher_outputs = {}
        with train_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = rec.get("id")
                if tid in subset_by_id and rec.get("teacher_output"):
                    teacher_outputs[tid] = rec["teacher_output"]
        logging.info(f"[{model_key}] teacher_outputs disponibles para subset: {len(teacher_outputs)}/{len(subset_by_id)}")

        # Reanudacion
        done_ids, prior_records = read_done(items_path)
        pending = [sid for sid in subset_by_id if sid in teacher_outputs and sid not in done_ids]
        if not pending:
            logging.info(f"[{model_key}] Ya completo ({len(prior_records)} items).")
            summary = summarize(prior_records, model_key, label)
            save_json(summary, eval_dir / f"{model_key}_results.json")
            continue

        logging.info(f"[{model_key}] pendientes: {len(pending)} (done: {len(done_ids)})")

        with items_path.open("a", encoding="utf-8") as items_f:
            for n, sid in enumerate(pending, 1):
                sample = subset_by_id[sid]
                question = sample["instruction"]
                reference = sample.get("reference", "")
                answer = teacher_outputs[sid]

                prompt = build_judge_prompt(
                    args.judge_prompt_version, question, answer, reference
                )
                t0 = time.time()
                scores = judge(model, tokenizer, device, prompt, args.judge_max_new_tokens)
                rec = {
                    "id": sid,
                    "instruction": question,
                    "answer": answer,
                    "reference": reference,
                    "scores": scores,
                    "weighted_score": weighted_score(scores),
                    "judge_seconds": round(time.time() - t0, 2),
                }
                items_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                items_f.flush()
                if n % 25 == 0 or n == len(pending):
                    logging.info(f"  [{model_key}] {n}/{len(pending)}")

        _, all_records = read_done(items_path)
        summary = summarize(all_records, model_key, label)
        save_json(summary, eval_dir / f"{model_key}_results.json")
        avg = summary["averages"]
        logging.info(f"[{model_key}] avg_weighted={avg.get('avg_weighted')} (n={summary['num_scored']})")

    comparison = rebuild_comparison(eval_dir)
    logging.info("=" * 60)
    logging.info("Comparativa final:")
    for mk, ms in comparison.get("models", {}).items():
        a = ms.get("averages", {})
        logging.info(
            f"  {ms.get('label', mk):<35} "
            f"acc={a.get('avg_accuracy', 0):.2f} rel={a.get('avg_relevance', 0):.2f} "
            f"com={a.get('avg_completeness', 0):.2f} cla={a.get('avg_clarity', 0):.2f} "
            f"w={a.get('avg_weighted', 0):.4f}"
        )
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
