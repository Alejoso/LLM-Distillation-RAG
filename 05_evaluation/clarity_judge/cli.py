import argparse
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone
from tqdm import tqdm

from .models import OllamaConfig, parse_input_item
from .client import OllamaClient
from .evaluator import Judge
from .utils import load_items, write_json, write_jsonl, summarize_results

def run_evaluation(args: argparse.Namespace):
    cfg = OllamaConfig(
        model=args.model,
        url=args.url,
        temperature=args.temperature,
        seed=args.seed,
        ollama_format=args.ollama_format,
    )

    client = OllamaClient(cfg)
    client.preflight_check()
    judge = Judge(client)

    items = load_items(args.input)
    results = []
    
    iterator = items if args.no_progress else tqdm(items, desc=f"Judging ({cfg.model})")

    for idx, raw in enumerate(iterator):
        row = {"id": raw.get("id", idx), "raw": raw}
        try:
            item = parse_input_item(raw)
            judged_data, error = judge.evaluate_single(item)
            if error:
                row["error"] = error
            else:
                row["scores"] = judged_data
        except Exception as e:
            row["error"] = str(e)
            
        results.append(row)

    save_outputs(args, results)

def save_outputs(args, results: List[dict]):
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    in_stem = Path(args.input).stem
    out_base = Path(args.outdir) / f"{in_stem}_{now}"
    
    if args.output_format == "jsonl":
        write_jsonl(f"{out_base}.jsonl", results)
    else:
        write_json(f"{out_base}.json", {"items": results})
        
    summary = summarize_results(results)
    write_json(f"{out_base}_summary.json", summary)
    
    print(f"Results: {out_base}")

def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(prog="clarity-judge")
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--model", default="llama3.1:latest")
    parser.add_argument("--url", default="http://localhost:11434/api/chat")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ollama-format", default="schema", choices=["schema", "json"])
    parser.add_argument("--output-format", default="jsonl", choices=["jsonl", "json"])
    parser.add_argument("--no-progress", action="store_true")
    
    args = parser.parse_args(argv)
    
    try:
        run_evaluation(args)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1