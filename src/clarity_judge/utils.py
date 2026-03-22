import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from .models import (
    CLARITY_LABELS, RELEVANCE_LABELS, ACCURACY_LABELS, COMPLETENESS_LABELS
)

def load_items(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    
    obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, list): return obj
    if isinstance(obj, dict) and "items" in obj: return obj["items"]
    if isinstance(obj, dict): return [obj]
    raise ValueError("Unsupported JSON format.")

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json(path: str, obj: Any):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = {
        "clarity": [], "relevance": [], "accuracy": [], "completeness": [], "weighted": []
    }
    counts = {
        "clarity": {}, "relevance": {}, "accuracy": {}, "completeness": {}
    }

    for r in results:
        s = r.get("scores")
        if not s: continue
        
        stats["clarity"].append(s["clarity_score"])
        stats["relevance"].append(s["relevance_score"])
        stats["accuracy"].append(s["accuracy_score"])
        stats["completeness"].append(s["completeness_score"])
        stats["weighted"].append(s["weighted_score"])
        
        counts["clarity"][s["clarity_label"]] = counts["clarity"].get(s["clarity_label"], 0) + 1
        counts["relevance"][s["relevance_label"]] = counts["relevance"].get(s["relevance_label"], 0) + 1
        counts["accuracy"][s["accuracy_label"]] = counts["accuracy"].get(s["accuracy_label"], 0) + 1
        counts["completeness"][s["completeness_label"]] = counts["completeness"].get(s["completeness_label"], 0) + 1

    avg = lambda x: round(sum(x) / len(x), 4) if x else 0
    
    return {
        "num_items": len(results),
        "num_scored": len(stats["weighted"]),
        "averages": {k: avg(v) for k, v in stats.items()},
        "distributions": {k: dict(sorted(v.items(), key=lambda x: -x[1])) for k, v in counts.items()}
    }