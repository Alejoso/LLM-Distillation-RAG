from typing import Annotated, Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError

CLARITY_LABELS = {1: "very unclear", 2: "unclear", 3: "mixed", 4: "clear", 5: "very clear"}
RELEVANCE_LABELS = {1: "off-topic", 2: "weakly relevant", 3: "partially answers", 4: "relevant", 5: "highly relevant"}
ACCURACY_LABELS = {1: "completely incorrect", 2: "mostly incorrect", 3: "partially correct", 4: "mostly correct", 5: "fully correct"}
COMPLETENESS_LABELS = {1: "very incomplete", 2: "mostly incomplete", 3: "partially complete", 4: "mostly complete", 5: "fully complete"}

JUDGE_WEIGHTS = {
    "accuracy": 0.4,
    "relevance": 0.3,
    "completeness": 0.2,
    "clarity": 0.1,
}

Score = Annotated[int, Field(ge=1, le=5)]

class SingleInput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: Optional[str] = None
    question: str
    answer: str
    reference: Optional[str] = None
    context: Optional[str] = None
    key_points: Optional[List[str]] = None

class SingleJudgeOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clarity_score: Score
    relevance_score: Score
    accuracy_score: Score
    completeness_score: Score
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    one_sentence_summary: str

    @field_validator("issues", "suggestions")
    @classmethod
    def max_three(cls, v: List[str]) -> List[str]:
        return v[:3]

    @field_validator("one_sentence_summary")
    @classmethod
    def single_sentence(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("one_sentence_summary must not be empty.")
        return v

class OllamaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    url: str = "http://localhost:11434/api/chat"
    timeout_s: int = 120
    max_retries: int = 4
    base_backoff_s: float = 0.8
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 42
    ollama_format: Literal["schema", "json"] = "schema"

    @field_validator("url")
    @classmethod
    def valid_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"url must start with http:// or https://, got: {v!r}")
        return v

def parse_input_item(raw: Dict[str, Any]) -> SingleInput:
    if "answer" in raw and "question" in raw:
        return SingleInput.model_validate(raw)
    raise ValueError("Item must have 'question' and 'answer'.")

SINGLE_SCHEMA = SingleJudgeOut.model_json_schema()