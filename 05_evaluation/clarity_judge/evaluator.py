from typing import Any, Dict, Optional, Tuple
from .models import (
    SingleInput, SingleJudgeOut, JUDGE_WEIGHTS, 
    CLARITY_LABELS, RELEVANCE_LABELS, ACCURACY_LABELS, COMPLETENESS_LABELS,
    SINGLE_SCHEMA
)
from .prompts import build_messages

class Judge:
    def __init__(self, client):
        self.client = client

    def evaluate_single(self, item: SingleInput) -> Tuple[Dict[str, Any], Optional[str]]:
        messages = build_messages(
            item.question, 
            item.answer, 
            reference=item.reference, 
            context=item.context, 
            key_points=item.key_points
        )
        
        fmt = SINGLE_SCHEMA if self.client.cfg.ollama_format == "schema" else "json"
        last_error = None

        for attempt in range(self.client.cfg.max_retries + 1):
            try:
                content = self.client.request(messages, fmt)
                obj = self.client.extract_json(content)
                parsed = SingleJudgeOut.model_validate(obj)
                
                return self._format_result(parsed), None
                
            except Exception as e:
                last_error = str(e)
                if attempt < self.client.cfg.max_retries:
                    self.client.sleep_backoff(attempt)
                    messages.append({
                        "role": "user", 
                        "content": "Invalid output. Return ONLY valid JSON with the required keys."
                    })
        
        return {}, last_error

    def _format_result(self, parsed: SingleJudgeOut) -> Dict[str, Any]:
        raw_scores = {
            "accuracy": int(parsed.accuracy_score),
            "relevance": int(parsed.relevance_score),
            "completeness": int(parsed.completeness_score),
            "clarity": int(parsed.clarity_score),
        }
        
        weighted = sum(JUDGE_WEIGHTS[dim] * raw_scores[dim] for dim in JUDGE_WEIGHTS)

        return {
            "clarity_score": raw_scores["clarity"],
            "clarity_label": CLARITY_LABELS[raw_scores["clarity"]],
            "relevance_score": raw_scores["relevance"],
            "relevance_label": RELEVANCE_LABELS[raw_scores["relevance"]],
            "accuracy_score": raw_scores["accuracy"],
            "accuracy_label": ACCURACY_LABELS[raw_scores["accuracy"]],
            "completeness_score": raw_scores["completeness"],
            "completeness_label": COMPLETENESS_LABELS[raw_scores["completeness"]],
            "weighted_score": round(weighted, 4),
            "issues": parsed.issues,
            "suggestions": parsed.suggestions,
            "one_sentence_summary": parsed.one_sentence_summary,
        }