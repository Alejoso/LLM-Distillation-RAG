"""Guardas deterministas para el LLM-as-Judge (pre-filtro fuera del modelo).

Motivacion
----------
El juez (Llama-2-7b-chat) NO puede detectar de forma confiable respuestas
degeneradas solo con prompt engineering: ante un candidato vacio, un eco de la
pregunta o relleno circular, el modelo *alucina* la respuesta correcta y asigna
5/5/5/5. Ese es exactamente el fallo que produjo el artefacto "student > teacher"
(un TinyLlama que emite salida truncada/degenerada recibe 5/5). Iterar el prompt
(v1 -> v2 -> v3) no lo arreglo; v3 incluso lo empeoro.

Solucion
--------
Antes de invocar al juez, se aplica un screen determinista de ALTA PRECISION:
solo dispara sobre clases inequivocas y nunca sobre una respuesta legitima
(una respuesta correcta contiene el hecho de la referencia, y ninguna guarda que
exige "el candidato no aporta el hecho" puede activarse sobre ella).

Reglas (en orden de prioridad):
  R0  abstention_match     ref dice "no existe/ no especifica" Y ans tambien
                           -> {5,5,5,5}  (abstencion honesta correcta)
  R1  non_answer           vacio / solo espacios / solo puntuacion o simbolos
                           -> {1,1,1,1}
  R2  abstention_mismatch  ans evade ("no se encuentra...") pero ref SI da el hecho
                           -> {1,2,1,4}  (~1.6 ponderado)
  R3  no_information        ans no aporta ningun token del hecho de la referencia
                           y no introduce contenido nuevo mas alla de la pregunta
                           (eco / circular / relleno) -> {1,2,1,2}  (~1.4 ponderado)

Si ninguna regla dispara, screen_answer devuelve None y el item pasa al juez LLM
(clases que requieren semantica real: confident-wrong, swapped, nonsense, etc.).

El modulo es puro (sin GPU / sin transformers) para poder testearlo en CI.
"""

from __future__ import annotations

import re
import unicodedata

# Pesos del juez (deben coincidir con calibrate_judge.JUDGE_WEIGHTS).
_WEIGHTS = {"accuracy": 0.4, "relevance": 0.3, "completeness": 0.2, "clarity": 0.1}

# --- Scores fijos por regla (elegidos para caer sobre el expected de cada clase) ---
_SCORES_NON_ANSWER = {"accuracy_score": 1, "relevance_score": 1, "completeness_score": 1, "clarity_score": 1}   # 1.0
_SCORES_ABSTENTION_OK = {"accuracy_score": 5, "relevance_score": 5, "completeness_score": 5, "clarity_score": 5} # 5.0
_SCORES_ABSTENTION_BAD = {"accuracy_score": 1, "relevance_score": 2, "completeness_score": 1, "clarity_score": 4} # 1.6
_SCORES_NO_INFO = {"accuracy_score": 1, "relevance_score": 2, "completeness_score": 1, "clarity_score": 2}        # 1.4

# Marcadores de abstencion (tras normalizar: minusculas + sin acentos).
_ABSTENTION_MARKERS = (
    "no se encuentra",
    "no especifica",
    "no se especifica",
    "no se menciona",
    "no se indica",
    "no tengo informacion",
    "no hay informacion",
    "no se dispone",
    "no dispongo",
    "no esta disponible",
    "no disponible",
    "informacion solicitada no",
    "fuera del corpus",
    "no figura",
    "no aparece en",
    "no se proporciona",
    "no cuento con",
)

# Palabras funcionales / genericas del dominio que NO cuentan como "contenido".
# El contenido real de una respuesta juridica es un nombre / numero / fecha /
# entidad, ninguno de los cuales aparece aqui.
_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo", "al", "del",
    "de", "en", "y", "o", "u", "a", "que", "se", "es", "su", "sus", "por",
    "para", "con", "como", "mas", "pero", "este", "esta", "estos", "estas",
    "ese", "esa", "dicho", "dicha", "mismo", "misma", "cual", "cuales", "cuanto",
    "cuantos", "cuanta", "cuantas", "que", "quien", "donde", "cuando",
    "segun", "sobre", "ademas", "tambien", "asi", "decir", "ser", "esto",
    "fue", "es", "sera", "esta", "estan", "hay", "tiene", "cual",
    # verbos / participios de relleno (eco y circular): describen la relacion
    # con la pregunta sin aportar el hecho. NO incluyen nombres propios ni
    # numeros, de modo que un hecho (aun equivocado) siempre sobrevive.
    "menciona", "mencionado", "mencionada", "mencionan", "mencionados",
    "establece", "establecido", "establecida", "corresponde", "designada",
    "designado", "refiere", "trata", "otorga", "otorgar", "otorgada",
    "otorgado", "recibe", "recibir", "impone", "encargada", "encargado",
    "aprobado", "aprobada", "aprobo", "aludido", "aludida", "referido",
    "referida", "indicado", "indicada", "senalado", "senalada", "dispuesto",
    "previsto", "contemplado", "respectivo", "respectiva", "valida", "valido",
    "requiere", "consultando", "original", "responder", "consultar",
    "articulo", "art", "ley", "leyes", "decreto", "numero", "num", "no", "si",
    "parrafo", "inciso", "literal", "capitulo", "titulo", "seccion",
    "fecha", "monto", "cifra", "entidad", "nombre", "ano", "cargo", "obligacion",
    "pena", "auxilio", "presupuesto", "ministerio", "documento", "texto",
    "corpus", "respuesta", "pregunta", "informacion", "solicitada",
}

_WORD_RE = re.compile(r"[0-9a-zñ.$/-]+")


def strip_accents(text: str) -> str:
    """Minusculas sin acentos (NFD -> descarta diacriticos)."""
    nfkd = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(text)).strip()


def _tokens(text: str) -> list:
    """Tokeniza conservando numeros/fechas/montos (contenido factual)."""
    norm = strip_accents(text)
    return [t for t in _WORD_RE.findall(norm) if t not in {".", "-", "/", "$"}]


def _content_tokens(text: str) -> set:
    """Tokens de contenido: quita stopwords y tokens de <=1 char sin digito."""
    out = set()
    for t in _tokens(text):
        core = t.strip(".$/-")
        if not core:
            continue
        if t in _STOPWORDS or core in _STOPWORDS:
            continue
        if len(core) <= 1 and not core.isdigit():
            continue
        out.add(core)
    return out


def _weighted(scores: dict) -> float:
    return round(sum(w * scores[f"{d}_score"] for d, w in _WEIGHTS.items()), 4)


def _is_abstention(text: str) -> bool:
    norm = _normalize(text)
    return any(m in norm for m in _ABSTENTION_MARKERS)


def _has_alnum(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def screen_answer(question: str, answer: str, reference: str = "") -> dict | None:
    """Aplica las guardas deterministas.

    Devuelve None si ninguna regla dispara (el item debe ir al juez LLM). Si una
    regla dispara, devuelve un dict con la misma forma que run_judge produce:
        {accuracy_score, relevance_score, completeness_score, clarity_score,
         one_sentence_summary, _guard: <rule_name>, _guard_weighted: <float>}
    """
    ans = answer or ""
    ref = reference or ""
    ans_stripped = ans.strip()

    ref_abstains = _is_abstention(ref)
    ans_abstains = _is_abstention(ans)

    # R0: abstencion honesta correcta (la referencia confirma la ausencia).
    if ref_abstains and ans_abstains:
        return _decision("abstention_match", _SCORES_ABSTENTION_OK,
                         "Both reference and answer correctly report the info is unavailable.")

    # R1: no-respuesta (vacio / whitespace / solo puntuacion o simbolos).
    if not ans_stripped or not _has_alnum(ans_stripped):
        return _decision("non_answer", _SCORES_NON_ANSWER,
                         "The answer is empty or contains no substantive content.")

    # R2: el candidato evade pero la referencia SI aporta el hecho.
    if ans_abstains and not ref_abstains and _content_tokens(ref):
        return _decision("abstention_mismatch", _SCORES_ABSTENTION_BAD,
                         "The answer evades although the reference provides the fact.")

    # R3: sin informacion nueva (eco / circular / relleno).
    q_content = _content_tokens(question)
    ref_content = _content_tokens(reference)
    ans_content = _content_tokens(ans)
    ref_distinctive = ref_content - q_content          # el hecho real de la referencia
    new_beyond_q = ans_content - q_content             # lo que el candidato aporta

    if ref_distinctive:  # solo si la referencia tiene un hecho identificable
        overlaps_fact = bool(ans_content & ref_distinctive)
        # El candidato no menciona ningun token del hecho y NO aporta ningun
        # contenido propio (solo eco de la pregunta + relleno) -> eco/circular.
        # El umbral estricto (== 0) es clave: una respuesta con un hecho
        # equivocado (p.ej. "Antioquia" cuando la ref dice "Panama") introduce
        # un token nuevo y por tanto NO dispara -> pasa al juez LLM.
        if not overlaps_fact and len(new_beyond_q) == 0:
            return _decision("no_information", _SCORES_NO_INFO,
                             "The answer restates the question without supplying the fact.")

    return None


def _decision(rule: str, scores: dict, summary: str) -> dict:
    out = dict(scores)
    out["one_sentence_summary"] = summary
    out["_guard"] = rule
    out["_guard_weighted"] = _weighted(scores)
    return out
