import re


def normalize_body(text: str, apply_body_rules: bool = True) -> str:
    # Base cleanup used for both metadata and body.
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)
    text = re.sub(r"[\uFFFD\uFFFE\uFFFF]", "", text)
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"[\t\r ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\(\s*\)", "", text)

    if not apply_body_rules:
        return text.strip()
    # Normalize spaced-out "A R T I C U L O" tokens.
    text = re.sub(
        r"\bA\s+R\s+T\s+[ÍI]\s+C\s+U\s+L\s+O\b",
        "ARTICULO",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bARTICULO\n(\d+°?)", r"ARTICULO \1", text)
    text = re.sub(r",\s*\n", ", ", text)
    # Normalize money formats like 2_50 or 0-04.
    text = re.sub(r"(\d)[_-](\d{2})\b", r"\1.\2", text)
    # Join numeric list headings (e.g., 1., 1.1, 1.2) with their content.
    text = re.sub(
        r"^\s*(\d+(?:\.\d+)*(?:[\.)])?)\s*\n+",
        r"\1 ",
        text,
        flags=re.MULTILINE,
    )
    # Merge punctuation-only and symbol-only lines with surrounding text.
    text = re.sub(r"([^\n])\n\s*([:;,.])\s*\n(\S)", r"\1\2 \3", text)
    text = re.sub(r"([^\n])\n\s*([:;,.])", r"\1\2", text)
    text = re.sub(r"\n\s*\$\s*\n", " $ ", text)
    text = re.sub(r"\bcon\s*\n\s*\$?\s*", "con $ ", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*\n\s*([A-Za-zÁÉÍÓÚÑáéíóúñ\.]+)\s*\n\s*\)", r"(\1)", text)
    text = re.sub(r"([A-Za-zÁÉÍÓÚÑáéíóúñ0-9])\s*\n\s*\)", r"\1)", text)
    # Join split decimal cents across lines (e.g., "0\n02" -> "0.02").
    text = re.sub(r"(\b\d+)\s*\n\s*(\d{2})\b", r"\1.\2", text)
    text = re.sub(
        r"([A-Za-zÁÉÍÓÚÑáéíóúñ]{2,})\s*\n\s*([A-Za-zÁÉÍÓÚÑáéíóúñ])\s*\n\s*([A-Za-zÁÉÍÓÚÑáéíóúñ]{2,})",
        r"\1\2\3",
        text,
    )
    # Merge ordinal marks that are split to the next line (e.g., "1\n°").
    text = re.sub(r"(\d+)\s*\n\s*([º°])", r"\1\2", text)
    # Remove standalone hyphen/quote lines and strip hyphen bullets in headers/signatures.
    text = re.sub(r"^\s*[-–—]+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\"+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*-\s*(?=[A-ZÁÉÍÓÚÑ])", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*-\s*(?=(?:El|La|Los|Las)\s)", " ", text)
    # Separate list items that are on the same line (e.g., "text; b)" -> "text;\nb)")
    text = re.sub(r"([;.])\s+([a-z]\))", r"\1\n\2", text)
    # Join roman numerals in parentheses with their content (e.g., "(i)\nText" -> "(i) Text").
    text = re.sub(
        r"^\s*\(([ivxlcdm]+)\)\s*\n+",
        r"(\1) ",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    # Join lowercase letters in parentheses with their content (e.g., "a.\nText" or "(a)\nText" -> "a. Text" or "(a) Text").
    text = re.sub(
        r"^\s*([a-z])\.\s*\n+",
        r"\1. ",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^\s*([a-z])\)\s*\n+",
        r"\1) ",
        text,
        flags=re.MULTILINE,
    )
    # Join number-only lines with the following text line (table-like lists).
    text = re.sub(
        r"^\s*(\d+)\s*\n([A-Za-zÁÉÍÓÚÑáéíóúñ])",
        r"\1 \2",
        text,
        flags=re.MULTILINE,
    )
    # Join amount-only lines with the previous text line.
    text = re.sub(
        r"([A-Za-zÁÉÍÓÚÑáéíóúñ].*)\n\s*(\d{1,3}(?:[\.,]\d{3})*(?:[\., ]\d{2})?)\s*$",
        r"\1 \2",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"([^\n])\n\s*(\d{1,3}(?:[\.,]\d{3})*(?:[\., ]\d{2})?)\s*$",
        r"\1 \2",
        text,
        flags=re.MULTILINE,
    )
    # If a new "De/Del" item is glued after an amount, break it onto a new line.
    text = re.sub(
        r"(\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d{2})?)\s+(De(?:l|\s+los|\s+las)?\s)",
        r"\1\n\2",
        text,
    )
    # Remove stray leading/trailing quotes per line.
    text = re.sub(r"^\s*\"+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\"+\s*$", "", text, flags=re.MULTILINE)
    # Join dates split across lines (e.g., "Septiembre\n9 de\n1890").
    text = re.sub(r"([A-Za-zÁÉÍÓÚÑáéíóúñ\.])\s*\n(\d+\s+de\b)", r"\1 \2", text)
    text = re.sub(r"\bde\s*\n(\d{4})\b", r"de \1", text)
    # Join lettered list markers (e.g., "b)") to the previous line.
    text = re.sub(r";\s*\n\s*([a-zA-Z]\))", r"; \1", text)
    # Join signature titles with names on the next line.
    text = re.sub(
        r"(^[^\n]{3,80}[,;:])\s*\n([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ]+)$",
        r"\1 \2",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"(^[^\n]{3,80}\.[ ]?)\s*\n([A-ZÁÉÍÓÚÑ]{2,})$",
        r"\1 \2",
        text,
        flags=re.MULTILINE,
    )
    # Join lines where a name continues on a lowercase line.
    text = re.sub(r"([A-Za-zÁÉÍÓÚÑáéíóúñ,;])\s*\n([a-záéíóúñ])", r"\1 \2", text)
    # Join all-caps names split across two short lines.
    text = re.sub(
        r"^([A-ZÁÉÍÓÚÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,}){0,2})\s*\n([A-ZÁÉÍÓÚÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,}){0,2})$",
        r"\1 \2",
        text,
        flags=re.MULTILINE,
    )
    # If a roman numeral line is followed by a single letter and a word, fold the letter into the word.
    text = re.sub(
        r"^([IVXLCDM])\s*\n([A-ZÁÉÍÓÚÑ])\s*\n([A-ZÁÉÍÓÚÑ]{2,})",
        r"\1\n\2\3",
        text,
        flags=re.MULTILINE,
    )
    # Remove whitespace-only lines before collapsing.
    text = re.sub(r"^\s+$", "", text, flags=re.MULTILINE)
    # Re-collapse empty lines introduced by the merges.
    text = re.sub(r"\n{2,}", "\n", text)
    # Remove punctuation-only lines that remain after merging.
    text = re.sub(r"^\s*[:;,.]\s*$", "", text, flags=re.MULTILINE)
    # Separate decree/resolution keywords from article headers
    text = re.sub(
        r"(DECRETA|RESUELVE|ORDENA|DISPONE):\s*(Artículo|ARTICULO|ART[ÍI]CULO)",
        r"\1:\n\2",
        text,
        flags=re.IGNORECASE,
    )
    # Join article and paragrafo headers with their content lines.
    # First, join "Artículo" on one line with the number on the next line
    text = re.sub(
        r"^(ART(?:[ÍI]CULO)?)\s*\n+(\d+(?:\s*\.?\s*[º°o])?\.?)",
        r"\1 \2",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    # Then join the complete article header with its content
    text = re.sub(
        r"^(ART(?:[ÍI]CULO)?\.?\s+(?:\d+(?:\s*\.?\s*[º°o])?|[IVXLCDM]+|primero|segundo|tercero|cuarto|quinto|sexto|s[eé]ptimo|octavo|noveno|d[eé]cimo|und[eé]cimo|duod[eé]cimo)\.?)(?:\s*\n+)",
        r"\1 ",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    text = re.sub(
        r"^(PAR[AÁ]GRAFO(?:\s+(?:\d+|primero|segundo|tercero|cuarto|quinto|sexto|s[eé]ptimo|octavo|noveno|d[eé]cimo))?(?:\s*\.?\s*[º°])?\.?\:?)\s*(?:\n+)",
        r"\1 ",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    return text.strip()
