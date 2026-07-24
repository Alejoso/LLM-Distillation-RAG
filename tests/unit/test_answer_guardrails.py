"""Tests de answer_guardrails (guarda determinista pre-LLM del juez).

Cubren:
  1. Cada regla dispara con el score correcto (non_answer, abstention_match,
     abstention_mismatch, no_information) usando strings reales de calibracion.
  2. PRECISION: la guarda NUNCA dispara sobre respuestas buenas ni sobre
     respuestas con un hecho equivocado-pero-sustantivo (esas van al juez LLM).
  3. El score ponderado de cada regla cae sobre el expected de su clase.
"""

import sys
from pathlib import Path

import pytest

_JUDGE_DIR = Path(__file__).resolve().parents[2] / "04_distillation" / "judge_calibration"
if str(_JUDGE_DIR) not in sys.path:
    sys.path.insert(0, str(_JUDGE_DIR))

from answer_guardrails import screen_answer, _weighted  # noqa: E402

W = {"accuracy": 0.4, "relevance": 0.3, "completeness": 0.2, "clarity": 0.1}


def _wt(g):
    return sum(W[d] * g[f"{d}_score"] for d in W)


# ---------------------------------------------------------------------------
# R1: no-respuesta (vacio / whitespace / puntuacion / simbolos) -> {1,1,1,1}
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("answer", ["", "   ", "...", "?", ".", "!!!", "-- --", "   \n\t "])
def test_non_answer_fires(answer):
    g = screen_answer("¿Qué fecha se menciona en el artículo 2?", answer, "23 de diciembre de 1993.")
    assert g is not None and g["_guard"] == "non_answer"
    assert _wt(g) == 1.0


# ---------------------------------------------------------------------------
# R0: abstencion correcta (ref confirma ausencia) -> {5,5,5,5}
# ---------------------------------------------------------------------------
def test_abstention_match_fires_five():
    g = screen_answer(
        "¿Cuál es la edad exacta del presidente del Congreso?",
        "El documento no especifica la edad del presidente del Congreso al momento de la aprobación.",
        "El documento no especifica la edad del presidente del Congreso.",
    )
    assert g is not None and g["_guard"] == "abstention_match"
    assert _wt(g) == 5.0


def test_abstention_match_corpus_variant():
    g = screen_answer(
        "¿Qué dice el artículo 23 de una ley que no existe?",
        "La información solicitada no se encuentra disponible en los documentos proporcionados.",
        "La información solicitada no se encuentra disponible en el corpus.",
    )
    assert g is not None and g["_guard"] == "abstention_match"


# ---------------------------------------------------------------------------
# R2: evasion cuando la referencia SI aporta el hecho -> ~1.6
# ---------------------------------------------------------------------------
def test_abstention_mismatch_fires_low():
    g = screen_answer(
        "¿Cuál es el nombre del presidente del Senado?",
        "La respuesta no se encuentra en el documento.",
        "Jorge Vélez.",
    )
    assert g is not None and g["_guard"] == "abstention_mismatch"
    assert abs(_wt(g) - 1.6) < 0.01


# ---------------------------------------------------------------------------
# R3: sin informacion (eco / circular / relleno) -> ~1.4
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,answer,reference", [
    # eco exacto de la pregunta
    ("¿Cuál es el monto del auxilio mencionado en el artículo 1?",
     "¿Cuál es el monto del auxilio mencionado en el artículo 1?",
     "$200.000 pesos anuales."),
    # pregunta reformulada como afirmacion, sin dar el dato
    ("¿Qué ministerio se menciona en el artículo 5 de la Ley 489 de 1998?",
     "El ministerio que se menciona en el artículo 5 de la Ley 489 de 1998.",
     "El Ministerio del Interior."),
    # circular
    ("¿Cuál es la fecha mencionada en el decreto?",
     "La fecha mencionada en el decreto es la fecha que se menciona en el decreto, es decir, la fecha del decreto mencionado.",
     "12 de octubre de 1985."),
    # relleno repetitivo
    ("¿Qué monto se aprobó en el artículo 1° de la Ley 62 de 1962?",
     "El monto aprobado en el artículo 1° de la Ley 62 de 1962 corresponde al monto aprobado por el artículo 1° de la mencionada Ley 62 de 1962.",
     "$48.274.487,80"),
])
def test_no_information_fires(question, answer, reference):
    g = screen_answer(question, answer, reference)
    assert g is not None and g["_guard"] == "no_information"
    assert abs(_wt(g) - 1.4) < 0.01


# ---------------------------------------------------------------------------
# PRECISION: la guarda NO debe disparar sobre estas clases (van al juez LLM)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,answer,reference", [
    # respuesta correcta terse
    ("¿Qué ciudad recibe auxilio en el Artículo 4?", "Pasto.", "Pasto."),
    # respuesta correcta completa
    ("¿Cuál es el nombre de la ley?", "La ley mencionada es la Ley 75 de 1936.", "Ley 75 de 1936."),
    # hecho EQUIVOCADO pero sustantivo (confident-wrong): debe ir al LLM, no forzarse a 1
    ("¿Qué Ministerio se menciona en el artículo 2?",
     "El Ministerio que se menciona en el artículo 2 es el Ministerio del Interior.",
     "El Ministerio de Justicia."),
    # numero equivocado (right_magnitude): al LLM
    ("¿Qué número de habitantes registra el censo?",
     "El censo registra 9.200.000 habitantes.", "8.500.000 habitantes."),
    # swapped facts: al LLM
    ("¿Cuántos años de cárcel establece la pena?",
     "El artículo establece una pena de quince años de cárcel.", "Veinte años."),
    # nonsense pero sustantivo (introduce contenido): al LLM
    ("¿Qué establece el artículo 7?",
     "El artículo 7 establece que los pingüinos deberán registrarse ante el Ministerio de Educación.",
     "El artículo 7 establece la finalidad del sistema penitenciario."),
])
def test_guardrail_does_not_fire_on_semantic_cases(question, answer, reference):
    assert screen_answer(question, answer, reference) is None


def test_correct_answer_never_forced_low():
    """Una respuesta que contiene el hecho de la referencia jamas dispara."""
    g = screen_answer("¿Cuál es el nombre del Ministerio en el artículo 3?",
                      "El Ministerio mencionado es el Ministerio de Hacienda y Crédito Público.",
                      "Ministerio de Hacienda y Crédito Público.")
    assert g is None


def test_weighted_helper_matches_manual():
    assert _weighted({"accuracy_score": 1, "relevance_score": 2,
                      "completeness_score": 1, "clarity_score": 4}) == 1.6
