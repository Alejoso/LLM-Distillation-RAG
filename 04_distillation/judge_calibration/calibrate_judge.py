#!/usr/bin/env python3
"""Calibracion del LLM-as-Judge sobre sets controlados.

Corre el juez (Llama-2-7b-chat-hf) sobre tres sets curados:
  - calibration_set.jsonl: items representativos con expected_scores.
  - traps_set.jsonl + traps_extended.jsonl: items adversariales.
  - holdout_set.jsonl: items independientes NO usados como few-shot en el
    prompt. Sirve para medir generalizacion sin riesgo de memorizacion.

Compara versiones del prompt (v1, v2, v3 por defecto). Cada version produce:
  - MAE global, por set y por dimension.
  - Bootstrap CI 95% sobre weighted MAE (n=60 es chico, los IC son criticos).
  - Conteo de false_5 (juez dio 5 cuando esperado <= 2) y false_low.
  - Breakdown por categoria.

Output:
  - results.jsonl con scores predichos por cada version + metadata.
  - report.json con metricas, IC, metadata (prompt hashes, seed, modelo).

Uso:
    python calibrate_judge.py --judge_name meta-llama/Llama-2-7b-chat-hf \\
        --versions v2 v3 --output_dir ./outputs
"""

import argparse
import hashlib
import json
import logging
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Permitir importar judge_prompts.py del mismo directorio
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_prompts import build_judge_prompt, get_system_prompt  # noqa: E402

# torch/transformers se importan dentro de main() para que el modulo se pueda
# importar en tests sin GPU/HF stack.

DIMENSIONS = ["accuracy_score", "relevance_score", "completeness_score", "clarity_score"]
JUDGE_WEIGHTS = {"accuracy": 0.4, "relevance": 0.3, "completeness": 0.2, "clarity": 0.1}
SUPPORTED_VERSIONS = ["v1", "v2", "v3"]
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260619


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Juez
# ---------------------------------------------------------------------------
def run_judge(model, tokenizer, device, prompt: str, max_new_tokens: int = 512) -> dict:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    if "[/INST]" in raw:
        raw = raw.split("[/INST]")[-1].strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"_parse_error": True, "_raw": raw[:500]}
    try:
        data = json.loads(match.group(0))
        return data
    except json.JSONDecodeError:
        return {"_parse_error": True, "_raw": raw[:500]}


def compute_weighted(scores: dict) -> float:
    total = 0.0
    for dim, weight in JUDGE_WEIGHTS.items():
        total += weight * scores.get(f"{dim}_score", 0)
    return round(total, 4)


def bootstrap_mae_ci(errors: list, n_iter: int = BOOTSTRAP_ITERATIONS,
                     seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> dict:
    """IC 95% para MAE via bootstrap. Devuelve {mean, lo, hi, n}."""
    if not errors:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    arr = np.asarray(errors, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_iter, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": round(float(arr.mean()), 4),
        "lo": round(float(lo), 4),
        "hi": round(float(hi), 4),
        "n": int(arr.size),
    }


# ---------------------------------------------------------------------------
# Metricas de calibracion
# ---------------------------------------------------------------------------
def analyze_predictions(records: list, version: str) -> dict:
    """Calcula metricas de agreement para una version del juez."""
    per_dim = {dim: {"errors": [], "exact": 0, "off_by_1": 0, "off_by_2plus": 0,
                     "false_5": 0, "false_low": 0} for dim in DIMENSIONS}
    weighted_errors = []
    parse_errors = 0

    for r in records:
        pred = r.get(f"pred_{version}", {})
        if pred.get("_parse_error"):
            parse_errors += 1
            continue
        expected = r["expected_scores"]

        for dim in DIMENSIONS:
            exp_val = expected[dim]
            pred_val = pred.get(dim)
            if pred_val is None or not isinstance(pred_val, (int, float)):
                continue
            diff = abs(pred_val - exp_val)
            per_dim[dim]["errors"].append(diff)
            if diff == 0:
                per_dim[dim]["exact"] += 1
            elif diff == 1:
                per_dim[dim]["off_by_1"] += 1
            else:
                per_dim[dim]["off_by_2plus"] += 1
            # falso 5: el juez dio 5 cuando esperado <= 2
            if pred_val == 5 and exp_val <= 2:
                per_dim[dim]["false_5"] += 1
            # falso bajo: juez dio <= 2 cuando esperado >= 4
            if pred_val <= 2 and exp_val >= 4:
                per_dim[dim]["false_low"] += 1

        pred_weighted = compute_weighted(pred)
        weighted_errors.append(abs(pred_weighted - r["expected_weighted"]))

    # Resumir
    summary = {"parse_errors": parse_errors, "n_evaluated": len(records) - parse_errors}
    for dim in DIMENSIONS:
        errors = per_dim[dim]["errors"]
        n = max(len(errors), 1)
        summary[dim] = {
            "mae": round(float(np.mean(errors)) if errors else 0.0, 3),
            "exact_pct": round(100 * per_dim[dim]["exact"] / n, 1),
            "off_by_1_pct": round(100 * per_dim[dim]["off_by_1"] / n, 1),
            "off_by_2plus_pct": round(100 * per_dim[dim]["off_by_2plus"] / n, 1),
            "false_5_count": per_dim[dim]["false_5"],
            "false_low_count": per_dim[dim]["false_low"],
        }
    summary["weighted_mae"] = round(float(np.mean(weighted_errors)) if weighted_errors else 0.0, 3)
    summary["weighted_mae_ci95"] = bootstrap_mae_ci(weighted_errors)
    for dim in DIMENSIONS:
        summary[dim]["mae_ci95"] = bootstrap_mae_ci(per_dim[dim]["errors"])
    return summary


def category_breakdown(records: list, version: str) -> dict:
    """Agrupa errores por categoria del item (e.g., confident_wrong)."""
    by_cat = {}
    for r in records:
        cat = r.get("category") or r.get("category_dummy") or "uncategorized"
        pred = r.get(f"pred_{version}", {})
        if pred.get("_parse_error"):
            continue
        exp_w = r["expected_weighted"]
        pred_w = compute_weighted(pred)
        by_cat.setdefault(cat, {"n": 0, "diffs": [], "exp_avg": [], "pred_avg": []})
        by_cat[cat]["n"] += 1
        by_cat[cat]["diffs"].append(pred_w - exp_w)
        by_cat[cat]["exp_avg"].append(exp_w)
        by_cat[cat]["pred_avg"].append(pred_w)
    out = {}
    for cat, d in by_cat.items():
        out[cat] = {
            "n": d["n"],
            "expected_weighted_avg": round(float(np.mean(d["exp_avg"])), 3),
            "judge_weighted_avg": round(float(np.mean(d["pred_avg"])), 3),
            "mean_signed_diff": round(float(np.mean(d["diffs"])), 3),
            "mean_abs_diff": round(float(np.mean([abs(x) for x in d["diffs"]])), 3),
        }
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def evaluate_set(model, tokenizer, device, items: list, versions: list, max_new: int) -> list:
    out_records = []
    n = len(items)
    for i, item in enumerate(items, 1):
        rec = dict(item)  # copy
        for v in versions:
            prompt = build_judge_prompt(
                v,
                item["instruction"],
                item["candidate_answer"],
                item.get("reference", ""),
            )
            t0 = time.time()
            pred = run_judge(model, tokenizer, device, prompt, max_new_tokens=max_new)
            rec[f"pred_{v}"] = pred
            rec[f"pred_{v}_seconds"] = round(time.time() - t0, 2)
        out_records.append(rec)
        if i % 5 == 0 or i == n:
            logging.info(f"  juez evaluado: {i}/{n}")
    return out_records


def _prompt_hash(version: str) -> str:
    """SHA-256 (12 chars) del system prompt para auditar cambios entre runs."""
    return hashlib.sha256(get_system_prompt(version).encode("utf-8")).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_name", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--calibration_set", default=None,
                        help="Path a calibration_set.jsonl (default: junto al script)")
    parser.add_argument("--traps_set", default=None,
                        help="Path a traps_set.jsonl (default: junto al script)")
    parser.add_argument("--traps_extended_set", default=None,
                        help="Path a traps_extended.jsonl (default: junto al script). "
                             "Usar '' para desactivarlo.")
    parser.add_argument("--holdout_set", default=None,
                        help="Path a holdout_set.jsonl (default: junto al script). "
                             "Usar '' para desactivarlo.")
    parser.add_argument("--output_dir", default="./outputs/judge_calibration",
                        help="Directorio donde guardar results y report")
    parser.add_argument("--versions", nargs="+", default=["v2", "v3"],
                        choices=SUPPORTED_VERSIONS,
                        help="Versiones del prompt a evaluar")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED,
                        help="Seed para bootstrap y generacion (do_sample=False igual fija decodificacion).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Si > 0, evaluar solo los primeros N items de cada set (debug)")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent

    def _resolve_optional_path(arg_value, default_name):
        if arg_value == "":
            return None
        return Path(arg_value or here / default_name)

    calibration_path = Path(args.calibration_set or here / "calibration_set.jsonl")
    traps_path = Path(args.traps_set or here / "traps_set.jsonl")
    traps_ext_path = _resolve_optional_path(args.traps_extended_set, "traps_extended.jsonl")
    holdout_path = _resolve_optional_path(args.holdout_set, "holdout_set.jsonl")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "calibrate_judge.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("=" * 60)
    logging.info(f"Calibracion del juez: {args.judge_name}")
    logging.info(f"versions: {args.versions}")
    logging.info("=" * 60)

    # Cargar sets
    cal_items = load_jsonl(calibration_path)
    trap_items = load_jsonl(traps_path)
    trap_ext_items = load_jsonl(traps_ext_path) if traps_ext_path and traps_ext_path.exists() else []
    holdout_items = load_jsonl(holdout_path) if holdout_path and holdout_path.exists() else []

    if args.limit > 0:
        cal_items = cal_items[: args.limit]
        trap_items = trap_items[: args.limit]
        trap_ext_items = trap_ext_items[: args.limit]
        holdout_items = holdout_items[: args.limit]

    logging.info(f"calibration items: {len(cal_items)} (de {calibration_path.name})")
    logging.info(f"trap items:        {len(trap_items)} (de {traps_path.name})")
    if trap_ext_items:
        logging.info(f"trap_extended:     {len(trap_ext_items)} (de {traps_ext_path.name})")
    if holdout_items:
        logging.info(f"holdout items:     {len(holdout_items)} (de {holdout_path.name})")

    # Marcar set para breakdown
    for it in cal_items:
        it["set"] = "calibration"
    for it in trap_items:
        it["set"] = "traps"
    for it in trap_ext_items:
        it["set"] = "traps_extended"
    for it in holdout_items:
        it["set"] = "holdout"

    # Metadata reproducible
    metadata = {
        "judge_name": args.judge_name,
        "versions": args.versions,
        "prompt_hashes": {v: _prompt_hash(v) for v in args.versions},
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "set_sizes": {
            "calibration": len(cal_items),
            "traps": len(trap_items),
            "traps_extended": len(trap_ext_items),
            "holdout": len(holdout_items),
        },
    }
    logging.info(f"prompt_hashes: {metadata['prompt_hashes']}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logging.info(f"Cargando juez {args.judge_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    logging.info("Juez cargado.")

    all_items = cal_items + trap_items + trap_ext_items + holdout_items
    records = evaluate_set(model, tokenizer, device, all_items, args.versions, args.max_new_tokens)

    # Anexar metadata por record (para auditoria fuera del report)
    for r in records:
        r["_meta"] = {
            "judge_name": metadata["judge_name"],
            "prompt_hashes": metadata["prompt_hashes"],
            "seed": metadata["seed"],
            "timestamp_utc": metadata["timestamp_utc"],
        }

    save_jsonl(records, output_dir / "results.jsonl")
    logging.info(f"results.jsonl guardado en {output_dir}")

    # Reporte
    report = {"metadata": metadata, "n_items": len(records)}
    set_names = ["calibration", "traps", "traps_extended", "holdout"]
    set_names = [s for s in set_names if any(r.get("set") == s for r in records)]
    for v in args.versions:
        overall = analyze_predictions(records, v)
        by_set = {}
        for set_name in set_names:
            subset = [r for r in records if r.get("set") == set_name]
            by_set[set_name] = analyze_predictions(subset, v)
        cats = category_breakdown(records, v)
        report[v] = {"overall": overall, "by_set": by_set, "by_category": cats}

    save_json(report, output_dir / "report.json")

    # Print resumen humano
    logging.info("=" * 60)
    logging.info("RESUMEN")
    logging.info("=" * 60)
    for v in args.versions:
        r = report[v]["overall"]
        ci = r.get("weighted_mae_ci95", {})
        logging.info(f"\n--- {v} (sha={metadata['prompt_hashes'][v]}) ---")
        logging.info(f"  parse_errors: {r['parse_errors']}/{len(records)}")
        ci_str = f" [95% CI {ci.get('lo')}..{ci.get('hi')}]" if ci.get("lo") is not None else ""
        logging.info(f"  weighted MAE: {r['weighted_mae']}{ci_str}")
        for dim in DIMENSIONS:
            d = r[dim]
            logging.info(
                f"  {dim:22s} MAE={d['mae']} exact={d['exact_pct']}% "
                f"false_5={d['false_5_count']} false_low={d['false_low_count']}"
            )
        # Resumen por set
        for set_name in set_names:
            s = report[v]["by_set"][set_name]
            sci = s.get("weighted_mae_ci95", {})
            sci_str = f" [{sci.get('lo')}..{sci.get('hi')}]" if sci.get("lo") is not None else ""
            logging.info(
                f"  set={set_name:15s} n={s['n_evaluated']:3d} "
                f"weighted_MAE={s['weighted_mae']}{sci_str}"
            )
    logging.info("\nreport.json escrito.")


if __name__ == "__main__":
    main()
