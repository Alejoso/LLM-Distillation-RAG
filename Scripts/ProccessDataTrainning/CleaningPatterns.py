PATTERNS = {
    # Law titles and references
    "law_title": r"^LEY\s+\d+\s+DE\s+\d{4}",
    "decree": r"DECRETO\s+\d+\s+DE\s+\d{4}",
    "date_parenthesis": r"\(\s*[a-záéíóúñ]+\s+\d{1,2}\s*\)",
    "decrees_line": r"DECRETA\s*:?\s*°*",
    "degree_symbol": r"°+",
    
    # Article numbering with strange characters
    "article_num": r"(?:Artículo|ART\.?|ARTICULO)\s+\d+(?:\.|°|º)?",
    "article_roman": r"\bART[ÍI]?CULO\s+[IVXLCDM]+\b",
    "chapter_roman": r"\bCAP[ÍI]TULO\s+[IVXLCDM]+\b",
    "title_roman": r"\bT[ÍI]TULO\s+[IVXLCDM]+\b",
    "transitory_provisions": r"\b[A-Z]\.\s*DISPOSICIONES\s+TRANSITORIAS\b",

    # Headers and legal noise
    "jurisprudence": r"\bJURISPRUDENCIA\b",
    "affects_validity": r"\bAfecta\s+la\s+vigencia\s+de:\s*",
    "previous_legislation": r"\bLEGISLACI[ÓO]N\s+ANTERIOR\b",

    # Literals and standalone numerals at line start
    "literal_parenthesis_start": r"^\s*\([a-z]\)\s*",
    "literal_upper_start": r"^\s*[;,]?\s*[A-Z]\.\s+",
    "literal_lower_start": r"^\s*[;,]?\s*[a-z]\)\s+",
    "ordinal_start": r"^\s*\d+o\.?\s+",
    "numeral_start": r"^\s*\d{1,3}\s+",
    "number_parenthesis_loose": r"\b\d{3,}\)\b",
    "parenthesis_loose": r"^\s*\)\s*$",
    "loose_or": r"^\s*o\s*$",

    # Signatures and abbreviations
    "signature_fdo": r"\(Fdo\.\),?\s*[A-ZÁÉÍÓÚÑ\s\.-]+",

    # Editorial text or appendices that don't contribute to MLM
    "to_be_transcribed": r"\(Para\s+ser\s+transcrito:\s*Se\s+adjunta\s+fotocopia\s+del\s+texto\s+íntegro\s+del\s+Instrumento\s+Internacional\s+mencionado\)\.?,?",
    "double_comma": r",,{1,}",
    
    # Sentence cleanup: references at start
    "ref_sentence_start": r"^(?:Art\.?|Artículo|Parágrafo|Numeral|Inciso)\s+\d+[°º.]*\s*",
    "list_start": r"^[a-z]\)\s+|^\d+\.\s+",
    
    # Cleanup: law references within sentences
    "ref_law_internal": r"(?:de\s+)?(?:la\s+)?(?:Ley|Decreto|Acto)\s+\d+\s+de\s+\d{4}",
    
    # Signatures and closures - REMOVE EVERYTHING FROM "DADO/DADA EN"
    "cut_from_given": r"Dad[ao]\s+en\b[\s\S]*\Z",
    
    # Position and title lines (president, secretary, etc.)
    "position_line": r"(?:El\s+)?(?:Presidente|Secretario|Subsecretario|Ministro|Vice\s*presidente|Procurador|Contralor|Defensor|Gobernador)\b.*?(?:\n|$)",
    
    # Executive sections and closures
    "executive_power": r"Poder\s+Ejecutivo\s*[^\n]*",
    "publish": r"Publíquese\s+y\s*(?:ejecútese|execútese).*?(?:\n|$)",
    "seal_ls": r"\(L\.\s*S\.\)",
    
    # URLs and emails
    "urls": r"https?://\S+",
    "emails": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    
    # Page numbers and cross-references
    "page": r"(?:pág|página|p\.)\s*\d+",
    "article_reference": r"\((?:numeral|artículo)\s*\d+(?:\.\d+)?\)",
}

# function to apply all cleaning patterns to text.
def apply_patterns(text):
    # Apply all cleaning patterns to text
    import re
    
    # First: remove EVERYTHING from "Dado/Dada en" to the end
    if "cut_from_given" in PATTERNS:
        pattern = PATTERNS["cut_from_given"]
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Then: apply the rest of the patterns
    for pattern_name, pattern in PATTERNS.items():
        if pattern_name != "cut_from_given":
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    
    return text
