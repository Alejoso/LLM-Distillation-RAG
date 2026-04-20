import json

import torch

from tests.helpers import (
    ExtremeTeacherModel,
    FakeTeacherModel,
    FakeTokenizer,
    write_json,
)


def test_prepare_teacher_prompt_uses_rag_when_enabled(distill_data_module, monkeypatch):
    monkeypatch.setattr(
        distill_data_module,
        "retrieve_contexts",
        lambda instruction, config: ["ctx A", "ctx B"],
    )

    sample = {"instruction": "Explain ML"}
    config = {"use_rag_for_teacher": True}

    prompt, contexts = distill_data_module.prepare_teacher_prompt(sample, config)

    assert contexts == ["ctx A", "ctx B"]
    assert "### Retrieved Context:" in prompt
    assert "ctx A" in prompt
    assert "ctx B" in prompt
    assert "### Instruction:" in prompt
    assert "### Response:" in prompt


def test_generate_distillation_jsonl_writes_expected_fields(distill_data_module, tmp_path, monkeypatch):
    processed_path = tmp_path / "processed.json"
    distilled_path = tmp_path / "train.jsonl"

    write_json(processed_path, [{"id": 1, "instruction": "What is AI?"}])

    monkeypatch.setattr(
        distill_data_module,
        "retrieve_contexts",
        lambda instruction, config: ["Artificial intelligence context"],
    )

    tokenizer = FakeTokenizer()
    teacher = FakeTeacherModel(tokenizer=tokenizer)

    config = {
        "processed_path": str(processed_path),
        "distilled_data_path": str(distilled_path),
        "max_len": 32,
        "max_new_tokens": 16,
        "use_rag_for_teacher": True,
    }

    distill_data_module.generate_distillation_jsonl(
        config=config,
        teacher=teacher,
        teacher_tokenizer=tokenizer,
        device=torch.device("cpu"),
    )

    lines = distilled_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    expected_keys = {
        "id",
        "instruction",
        "prompt",
        "used_rag",
        "retrieved_contexts",
        "teacher_output",
        "logits",
    }

    assert expected_keys.issubset(record.keys())
    assert record["id"] == 1
    assert record["instruction"] == "What is AI?"
    assert record["used_rag"] is True
    assert record["retrieved_contexts"] == ["Artificial intelligence context"]
    assert isinstance(record["teacher_output"], str)
    assert isinstance(record["logits"], list)
    assert len(record["logits"]) == 32


def test_generate_distillation_jsonl_clamps_logits_and_replaces_empty_response(distill_data_module, tmp_path, monkeypatch):
    processed_path = tmp_path / "processed.json"
    distilled_path = tmp_path / "train.jsonl"

    write_json(processed_path, [{"id": 2, "instruction": "Explain overfitting"}])

    monkeypatch.setattr(
        distill_data_module,
        "retrieve_contexts",
        lambda instruction, config: [],
    )

    tokenizer = FakeTokenizer()
    teacher = ExtremeTeacherModel(tokenizer=tokenizer)

    config = {
        "processed_path": str(processed_path),
        "distilled_data_path": str(distilled_path),
        "max_len": 16,
        "max_new_tokens": 8,
        "use_rag_for_teacher": False,
    }

    distill_data_module.generate_distillation_jsonl(
        config=config,
        teacher=teacher,
        teacher_tokenizer=tokenizer,
        device=torch.device("cpu"),
    )

    record = json.loads(distilled_path.read_text(encoding="utf-8").strip())
    assert record["teacher_output"] == "[EMPTY_RESPONSE]"

    logits = torch.tensor(record["logits"], dtype=torch.float32)
    assert torch.max(logits).item() <= 10.0
    assert torch.min(logits).item() >= -10.0
