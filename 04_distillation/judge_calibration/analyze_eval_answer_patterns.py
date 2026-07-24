#!/usr/bin/env python3
"""Sanity check sobre las respuestas YA generadas en los eval items (1500/modelo).

Cuenta, por modelo, cuantas respuestas caen en patrones que el juez v1 NO
detecta y v2/v3 si: empty/punctuation-only, idioma equivocado dominante,
eco de la pregunta, filler circular, truncado a media palabra, repeticion de
bigramas, salida muy corta.

El objetivo es documentar CUANTO del resultado "student > teacher" en el eval
del 2026-06-19 puede explicarse por el sesgo del juez v1 sobre estos patrones.

Uso:
    python analyze_eval_answer_patterns.py \\
        --eval_dir /path/to/logs/20260619/extracted/eval \\
        --out_path /path/to/sanity_report.json

El script lee los archivos *_items.jsonl (base_student, no_rag, with_rag,
teacher_no_rag, teacher_with_rag) e imprime una tabla.
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

EXPECTED_FILES = [
    "base_student_items.jsonl",
    "no_rag_items.jsonl",
    "with_rag_items.jsonl",
    "teacher_no_rag_items.jsonl",
    "teacher_with_rag_items.jsonl",
]

# Pequena lista de stopwords inglesas usadas como heuristica de idioma.
ENGLISH_MARKERS = {
    "the", "of", "and", "is", "are", "was", "were", "to", "in", "for",
    "with", "according", "article", "law", "decree", "this", "that",
    "which", "from", "by", "on", "as", "be", "has", "have", "shall",
}
SPANISH_MARKERS = {
    "el", "la", "los", "las", "de", "del", "que", "se", "y", "en",
    "por", "para", "con", "como", "es", "son", "fue", "según", "segun",
    "ley", "decreto", "articulo", "artículo", "esta", "este", "ministerio",
}


def normalize(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text).lower()


def is_empty_or_punctuation(answer: str) -> bool:
    stripped = answer.strip()
    if not stripped:
        return True
    if len(stripped) <= 5 and re.fullmatch(r"[\.,;:!\?\-\s]+", stripped):
        return True
    return False


def english_dominant(answer: str) -> bool:
    """Heuristica: la respuesta tiene >= 8 marcadores en ingles y al menos 2x mas que en espanol."""
    tokens = re.findall(r"[a-záéíóúñü]+", normalize(answer))
    if not tokens:
        return False
    en = sum(1 for t in tokens if t in ENGLISH_MARKERS)
    es = sum(1 for t in tokens if t in SPANISH_MARKERS)
    return en >= 8 and en >= 2 * max(es, 1)


def language_mixed(answer: str) -> bool:
    """English + Spanish mezclados en la misma respuesta."""
    tokens = re.findall(r"[a-záéíóúñü]+", normalize(answer))
    if len(tokens) < 10:
        return False
    en = sum(1 for t in tokens if t in ENGLISH_MARKERS)
    es = sum(1 for t in tokens if t in SPANISH_MARKERS)
    return en >= 4 and es >= 4


def echo_of_question(question: str, answer: str) -> bool:
    """La respuesta contiene >70% de la pregunta literal y aporta poco mas."""
    q = normalize(question).strip().rstrip("?¿").strip()
    a = normalize(answer)
    if len(q) < 15 or len(a) == 0:
        return False
    # Heuristica: si una subcadena de >=70% de la pregunta aparece literal en la respuesta
    # y la respuesta no tiene mucho mas largo, es un echo.
    if q in a and len(a) <= int(1.5 * len(q)):
        return True
    return False


def circular_filler(question: str, answer: str) -> bool:
    """La respuesta repite frases con sustantivos clave de la pregunta y no aporta cifras/nombres concretos."""
    q_words = set(w for w in re.findall(r"\w+", normalize(question)) if len(w) >= 4)
    a_norm = normalize(answer)
    if len(a_norm) < 40:
        return False
    bigrams = Counter()
    a_tokens = re.findall(r"\w+", a_norm)
    for i in range(len(a_tokens) - 1):
        bigrams[(a_tokens[i], a_tokens[i + 1])] += 1
    # Repetir el mismo bigrama 3+ veces sugiere filler
    repeated = sum(1 for _, c in bigrams.most_common(5) if c >= 3)
    # Ademas: la mayoria de las palabras de contenido son las de la pregunta
    overlap = sum(1 for t in a_tokens if t in q_words)
    overlap_ratio = overlap / max(len(a_tokens), 1)
    return repeated >= 2 and overlap_ratio >= 0.25


def truncated_midword(answer: str) -> bool:
    """Termina sin puntuacion y la ultima palabra parece truncada (sin sentence end)."""
    s = answer.rstrip()
    if not s:
        return False
    if s[-1] in ".!?\")”’":
        return False
    last = s.split()[-1] if s.split() else ""
    return len(last) >= 2 and last.isalpha()


def too_short(answer: str, min_chars: int = 12) -> bool:
    return 0 < len(answer.strip()) < min_chars


def analyze_file(path: Path) -> dict:
    counts = Counter()
    weighted_by_pattern = {k: [] for k in [
        "empty_or_punctuation", "english_dominant", "language_mixed",
        "echo_of_question", "circular_filler", "truncated_midword",
        "too_short", "clean",
    ]}
    total = 0
    weighted_total = []
    for line in path.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        total += 1
        ans = d.get("answer", "") or ""
        q = d.get("instruction", "") or ""
        w = d.get("weighted_score")
        if isinstance(w, (int, float)):
            weighted_total.append(w)

        flagged = False
        for name, pred in [
            ("empty_or_punctuation", is_empty_or_punctuation(ans)),
            ("english_dominant", english_dominant(ans)),
            ("language_mixed", language_mixed(ans)),
            ("echo_of_question", echo_of_question(q, ans)),
            ("circular_filler", circular_filler(q, ans)),
            ("truncated_midword", truncated_midword(ans)),
            ("too_short", too_short(ans)),
        ]:
            if pred:
                counts[name] += 1
                if isinstance(w, (int, float)):
                    weighted_by_pattern[name].append(w)
                flagged = True
        if not flagged:
            counts["clean"] += 1
            if isinstance(w, (int, float)):
                weighted_by_pattern["clean"].append(w)

    def avg(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    return {
        "file": path.name,
        "n_total": total,
        "weighted_avg_all": avg(weighted_total),
        "patterns": {
            name: {
                "count": counts[name],
                "pct": round(100 * counts[name] / max(total, 1), 1),
                "weighted_avg_when_flagged": avg(weighted_by_pattern[name]),
            }
            for name in weighted_by_pattern
        },
    }


def print_table(reports: list):
    pattern_keys = ["empty_or_punctuation", "english_dominant", "language_mixed",
                    "echo_of_question", "circular_filler", "truncated_midword",
                    "too_short", "clean"]
    header = ["model", "n", "w_avg"] + [k[:14] for k in pattern_keys]
    widths = [max(len(h), 16) for h in header]
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in reports:
        row = [
            r["file"].replace("_items.jsonl", "")[:16],
            str(r["n_total"]),
            str(r["weighted_avg_all"]),
        ]
        for k in pattern_keys:
            p = r["patterns"][k]
            cell = f"{p['count']} ({p['pct']}%)"
            row.append(cell)
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))

    print("\nweighted_avg_when_flagged (juez v1 vs cada patron):")
    sub_header = ["model"] + pattern_keys
    sub_widths = [max(len(h), 22) for h in sub_header]
    print(" | ".join(h.ljust(w) for h, w in zip(sub_header, sub_widths)))
    print("-+-".join("-" * w for w in sub_widths))
    for r in reports:
        row = [r["file"].replace("_items.jsonl", "")[:22]]
        for k in pattern_keys:
            v = r["patterns"][k]["weighted_avg_when_flagged"]
            row.append("—" if v is None else f"{v}")
        print(" | ".join(c.ljust(w) for c, w in zip(row, sub_widths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True,
                        help="Carpeta con *_items.jsonl (e.g. logs/20260619/extracted/eval)")
    parser.add_argument("--out_path", default=None,
                        help="Si se da, escribe el JSON con el reporte agregado.")
    args = parser.parse_args()
    eval_dir = Path(args.eval_dir).resolve()

    reports = []
    for fname in EXPECTED_FILES:
        path = eval_dir / fname
        if not path.exists():
            print(f"[skip] {path.name}: no existe")
            continue
        reports.append(analyze_file(path))

    print_table(reports)

    if args.out_path:
        out = Path(args.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(reports, indent=2, ensure_ascii=False))
        print(f"\nReporte JSON: {out}")


if __name__ == "__main__":
    main()
