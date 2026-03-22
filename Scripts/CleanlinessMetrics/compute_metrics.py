import re
from typing import Dict
import sys
from pathlib import Path

# Calculates the ratio of lines with 3 or fewer characters
def short_lines_ratio(text: str) -> float:
    lines = [l for l in text.split("\n") if l.  strip()]
    if not lines:
        return 0.0

    short = sum(1 for l in lines if len(l.strip()) <= 3)
    return short / len(lines)

# Calculates the ratio of fragmented words (words separated by spaces)
def fragmented_words_ratio(text: str) -> float:
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0.0

    fragmented = re.findall(r"\b(\w\s){2,}\w\b", text)
    return len(fragmented) / len(words)

# Validates the integrity of legal headers and structured elements
def header_integrity_ratio(text: str) -> float:

    header_pattern = (
        r"^(?:"
        # ARTICULO 1 / ARTÍCULO 1° / ART. 1
        r"ART(?:[ÍI]CULO)?\.?\s+"
        r"(?:\d+(?:\s*[º°o])?|[IVXLCDM]+|"
        r"primero|segundo|tercero|cuarto|quinto|sexto|"
        r"s[eé]ptimo|octavo|noveno|d[eé]cimo|"
        r"und[eé]cimo|duod[eé]cimo)"
        r"\.?"
        r"|"
        # CAPITULO I / CAPÍTULO
        r"CAP[IÍ]TULO\s+(?:[IVXLCDM]+|[0-9]+|primero|segundo|tercero)"
        r"|"
        # PARAGRAFO 1 / PARÁGRAFO primero
        r"PAR[AÁ]GRAFO(?:\s+"
        r"(?:\d+|primero|segundo|tercero|cuarto|quinto|"
        r"sexto|s[eé]ptimo|octavo|noveno|d[eé]cimo)"
        r")?"
        r"|"
        # SECCIÓN I / SECCION
        r"SECCI[OÓ]N\s+(?:[IVXLCDM]+|[0-9]+)"
        r"|"
        # TÍTULO I / TITULO
        r"T[ÍI]TULO\s+(?:[IVXLCDM]+|[0-9]+)"
        r"|"
        # DISPOSICIÓN / DISPOSICIÓN GENERAL
        r"DISPOSICI[OÓ]N(?:\s+\w+)?"
        r"|"
        # TRANSITORIO / TRANSITORIOS
        r"TRANSITORIO(?:\s+[0-9]+|S)?"
        r"|"
        # Listas numéricas 1. 1.1 1)
        r"\d+(?:\.\d+)*[\.)]"
        r"|"
        # Listas alfabéticas a) b) A) B)
        r"[a-zA-Z]\)"
        r"|"
        # Romanas (i) (ii) (I) (II)
        r"\([ivxlcdmIVXLCDM]+\)"
        r"|"
        # INCISO / LITERAL
        r"(?:INCISO|LITERAL)\s+[a-z]"
        r"|"
        # NUMERAL / PUNTO
        r"(?:NUMERAL|PUNTO)\s+\d+"
        r")"
    )

    headers = re.findall(header_pattern, text, flags=re.MULTILINE | re.IGNORECASE)

    if not headers:
        return 0.0

    # Validar que no estén fragmentados
    valid = 0
    for h in headers:
        if "\n" not in h and len(h.strip()) > 2:
            valid += 1

    return valid / len(headers)


# Assigns a score based on the ratio of short lines
def score_lines(ratio: float) -> int:
    if ratio == 0:
        return 30
    elif ratio < 0.03:
        return 25
    elif ratio < 0.07:
        return 15
    else:
        return 0


# Assigns a score based on the fragmentation ratio of words
def score_fragmentation(ratio: float) -> int:
    if ratio == 0:
        return 30
    elif ratio < 0.005:
        return 20
    elif ratio < 0.02:
        return 10
    else:
        return 0


# Assigns a score based on the structural integrity ratio of headers
def score_structure(ratio: float) -> int:
    if ratio >= 1.0:
        return 25
    elif ratio >= 0.9:
        return 20
    elif ratio >= 0.75:
        return 10
    else:
        return 0


# Classifies a score into quality categories (HIGH, MEDIUM, LOW, DEFECTIVE)
def classify_score(score: int) -> str:
    if score >= 85:
        return "HIGH"
    elif score >= 70:
        return "MEDIUM"
    elif score >= 50:
        return "LOW"
    else:
        return "DEFECTIVE"


# Computes overall quality score and returns metrics with individual ratios and classification
def compute_quality_score(text: str) -> Dict:

    line_ratio = short_lines_ratio(text)
    frag_ratio = fragmented_words_ratio(text)
    header_ratio = header_integrity_ratio(text)

    total_score = (
        int(score_lines(line_ratio) * 45 / 30) +
        int(score_fragmentation(frag_ratio) * 45 / 30) +
        int(score_structure(header_ratio) * 10 / 25)
    )

    return {
        "line_ratio": round(line_ratio, 4),
        "fragmented_ratio": round(frag_ratio, 4),
        "header_integrity": round(header_ratio, 4),
        "quality_score": total_score,
        "quality_status": classify_score(total_score),
        "version": "V1"
    }

# Example usage: python compute_metrics.py path/to/document.txt
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python compute_metrics.py file.txt")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print("File not found.")
        sys.exit(1)

    text = file_path.read_text(encoding="utf-8")
    result = compute_quality_score(text)

    print("\nQUALITY REPORT")
    for k, v in result.items():
        print(f"{k}: {v}")