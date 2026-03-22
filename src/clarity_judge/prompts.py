from typing import List, Optional, Dict

def get_system_base() -> str:
    return (
        "You are a strict evaluator of FOUR dimensions: CLARITY, RELEVANCE, ACCURACY, and COMPLETENESS.\n"
        "IMPORTANT:\n"
        "- Judge ONLY these four dimensions, do NOT evaluate style or length by itself.\n"
        "- Do NOT reward confidence, fancy wording, or length by itself.\n"
        "- If incorrectness directly harms relevance, reflect that in relevance.\n"
        "- Write issues/suggestions/summary in the SAME language as the question.\n"
        "Output ONLY valid JSON. No markdown. No extra text.\n"
    )

def get_scoring_instructions() -> str:
    return (
        "CLARITY score (1-5): 1 very unclear to 5 very clear.\n\n"
        "RELEVANCE score (1-5): 1 off-topic to 5 directly answers all parts.\n\n"
        "ACCURACY score (1-5): 1 completely incorrect to 5 fully correct.\n\n"
        "COMPLETENESS score (1-5): 1 very incomplete to 5 fully complete.\n\n"
        "Limits: issues <= 3, suggestions <= 3.\n"
        "one_sentence_summary must be exactly one sentence.\n"
    )

def build_messages(
    question: str,
    answer: str,
    reference: Optional[str] = None,
    context: Optional[str] = None,
    key_points: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    user_content = f"Question:\n{question}\n"
    if context:
        user_content += f"\nContext:\n{context}\n"
    if reference:
        user_content += f"\nReference answer:\n{reference}\n"
    if key_points:
        user_content += f"\nKey points expected:\n{', '.join(key_points)}\n"
    user_content += f"\nAnswer to evaluate:\n{answer}"

    return [
        {"role": "system", "content": get_system_base()},
        {"role": "system", "content": get_scoring_instructions()},
        {
            "role": "system",
            "content": (
                "Return ONLY JSON with these exact keys:\n"
                "clarity_score, relevance_score, accuracy_score, completeness_score, "
                "issues, suggestions, one_sentence_summary"
            ),
        },
        {"role": "user", "content": user_content},
    ]