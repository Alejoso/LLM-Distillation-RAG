"""Tests del modulo judge_prompts (sin GPU, sin HF).

Verifican:
  1. Las 3 versiones se construyen y todas siguen el formato Llama-2 [INST].
  2. v1, v2 ponen el reference antes que el answer; v3 lo invierte.
  3. v3 incluye etiquetas anti-leak (ANSWER_TO_EVALUATE, GROUND_TRUTH) que
     refuerzan no atribuir contenido de la referencia al candidato.
  4. v3 tiene reglas explicitas para empty/circular y para honest-uncertainty.
  5. build_judge_prompt rechaza versiones desconocidas.
  6. get_system_prompt devuelve los textos correspondientes.
"""

import sys
from pathlib import Path

import pytest

_JUDGE_DIR = Path(__file__).resolve().parents[2] / "04_distillation" / "judge_calibration"
if str(_JUDGE_DIR) not in sys.path:
    sys.path.insert(0, str(_JUDGE_DIR))

from judge_prompts import (  # noqa: E402
    JUDGE_SYSTEM_PROMPT_V1,
    JUDGE_SYSTEM_PROMPT_V2,
    JUDGE_SYSTEM_PROMPT_V3,
    JUDGE_SYSTEM_PROMPT_V4,
    build_judge_prompt,
    get_system_prompt,
)


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
def test_builds_llama2_inst_format(version):
    prompt = build_judge_prompt(version, "P?", "A", reference="R")
    assert prompt.startswith("<s>[INST] <<SYS>>")
    assert "<</SYS>>" in prompt
    assert prompt.rstrip().endswith("[/INST]")


@pytest.mark.parametrize("version,system", [
    ("v1", JUDGE_SYSTEM_PROMPT_V1),
    ("v2", JUDGE_SYSTEM_PROMPT_V2),
    ("v3", JUDGE_SYSTEM_PROMPT_V3),
])
def test_get_system_prompt(version, system):
    assert get_system_prompt(version) is system


def test_unknown_version_raises():
    with pytest.raises(ValueError):
        build_judge_prompt("v99", "P", "A", reference="R")
    with pytest.raises(ValueError):
        get_system_prompt("foo")


@pytest.mark.parametrize("version", ["v1", "v2", "v4"])
def test_reference_before_answer(version):
    """v1/v2/v4 usan orden reference-first (empiricamente mejor que v3)."""
    prompt = build_judge_prompt(version, "Q?", "MY_ANSWER", reference="MY_REFERENCE")
    ref_idx = prompt.index("MY_REFERENCE")
    ans_idx = prompt.index("MY_ANSWER")
    assert ref_idx < ans_idx, f"{version}: reference debe ir antes que answer"


def test_v4_targets_off_topic_and_absurd():
    """v4 debe cubrir las clases que solo la semantica resuelve."""
    sp = JUDGE_SYSTEM_PROMPT_V4.lower()
    assert "off-topic" in sp or "off topic" in sp
    assert "absurd" in sp or "hallucinated" in sp
    assert "never state" in sp or "never states" in sp or "without ever stating" in sp


def test_v3_answer_before_reference():
    prompt = build_judge_prompt("v3", "Q?", "MY_ANSWER", reference="MY_REFERENCE")
    ans_idx = prompt.index("MY_ANSWER")
    ref_idx = prompt.index("MY_REFERENCE")
    assert ans_idx < ref_idx, "v3: answer debe ir ANTES que reference para reducir leak"


def test_v3_uses_anti_leak_labels():
    prompt = build_judge_prompt("v3", "Q?", "A", reference="R")
    assert "[ANSWER_TO_EVALUATE]" in prompt
    assert "[GROUND_TRUTH]" in prompt
    assert "do NOT enrich" in prompt or "do NOT enrich it" in prompt


def test_v3_has_empty_circular_rule():
    """v3 debe tener regla explicita para respuestas vacias/circulares/echo."""
    sp = JUDGE_SYSTEM_PROMPT_V3.lower()
    assert "empty" in sp and "circular" in sp
    assert "echo" in sp or "restates the question" in sp or "repeats the question" in sp


def test_v3_distinguishes_4a_4b():
    """Honest-uncertainty: 4a (premiar) vs 4b (penalizar) deben estar claras."""
    sp = JUDGE_SYSTEM_PROMPT_V3
    assert "4a" in sp and "4b" in sp


def test_v3_neutral_on_length():
    """v3 NO debe penalizar respuestas correctas largas (sesgo contra teacher)."""
    sp = JUDGE_SYSTEM_PROMPT_V3.lower()
    assert "length is neutral" in sp or "do not penalize a correct answer for being verbose" in sp


def test_v3_examples_cover_traps():
    """v3 debe incluir Example F (empty), G (circular) y H (evasive)."""
    sp = JUDGE_SYSTEM_PROMPT_V3
    assert "Example F" in sp
    assert "Example G" in sp
    assert "Example H" in sp


def test_empty_reference_omits_block():
    """Si reference vacio, el bloque user-content no debe aparecer.

    Nota: en v3 las EXAMPLES del system prompt SI mencionan '[GROUND_TRUTH]'.
    Lo que NO debe aparecer es el bloque de user content '[GROUND_TRUTH] (use ONLY...'.
    """
    prompt = build_judge_prompt("v1", "Q?", "A", reference="")
    user_part = prompt.split("<</SYS>>")[-1]
    assert "Reference answer:" not in user_part

    prompt = build_judge_prompt("v3", "Q?", "A", reference="")
    user_part = prompt.split("<</SYS>>")[-1]
    assert "[GROUND_TRUTH] (use ONLY" not in user_part


def test_prompts_have_required_json_keys_instruction():
    """Las 3 versiones deben pedir las 5 claves del JSON de salida."""
    for sp in (JUDGE_SYSTEM_PROMPT_V1, JUDGE_SYSTEM_PROMPT_V2, JUDGE_SYSTEM_PROMPT_V3):
        for key in [
            "accuracy_score",
            "relevance_score",
            "completeness_score",
            "clarity_score",
            "one_sentence_summary",
        ]:
            assert key in sp, f"falta '{key}' en uno de los prompts"


def test_v3_no_regression_on_critical_rules_from_v2():
    """v3 debe mantener: confident-wrong penalizado, language mismatch, hallucinations."""
    sp = JUDGE_SYSTEM_PROMPT_V3.lower()
    assert "hallucinated" in sp
    assert "different language" in sp or "language mismatch" in sp.replace("-", " ")
    assert "confident" in sp


def test_question_appears_in_prompt():
    """La pregunta siempre va en el prompt, en todas las versiones."""
    for v in ["v1", "v2", "v3", "v4"]:
        p = build_judge_prompt(v, "PREGUNTA_UNICA_X9Z", "A", reference="R")
        assert "PREGUNTA_UNICA_X9Z" in p
