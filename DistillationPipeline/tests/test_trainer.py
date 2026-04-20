import pytest
import torch

from src.trainer import train_student
from tests.helpers import (
    FakeStudentModel,
    FakeTokenizer,
    make_distilled_record,
    write_jsonl,
)


def test_train_student_runs_writes_logs_and_reloads_checkpoint(tmp_path, capsys):
    max_len = 24
    distilled_path = tmp_path / "train.jsonl"
    checkpoint_path = tmp_path / "last.pt"
    log_file = tmp_path / "train.log"
    final_model_dir = tmp_path / "final_model"

    write_jsonl(
        distilled_path,
        [make_distilled_record(max_len=max_len, vocab_size=256, used_rag=False)],
    )

    config = {
        "distilled_data_path": str(distilled_path),
        "max_len": max_len,
        "batch_size": 1,
        "lr": 1e-3,
        "checkpoint_path": str(checkpoint_path),
        "log_file": str(log_file),
        "final_model_dir": str(final_model_dir),
        "epochs": 1,
        "alpha": 0.7,
        "temperature": 2.0,
        "save_steps": 1,
    }

    student = FakeStudentModel(vocab_size=256)
    tokenizer = FakeTokenizer(vocab_size=256)
    device = torch.device("cpu")

    train_student(config, student, tokenizer, device)
    assert checkpoint_path.exists()
    assert log_file.exists()
    assert final_model_dir.exists()
    assert (final_model_dir / "pytorch_model.bin").exists()
    assert (final_model_dir / "tokenizer.json").exists()

    student_reloaded = FakeStudentModel(vocab_size=256)
    train_student(config, student_reloaded, tokenizer, device)
    captured = capsys.readouterr()
    assert "Checkpoint loaded." in captured.out


def test_train_student_raises_on_vocab_mismatch(tmp_path):
    max_len = 24
    distilled_path = tmp_path / "train.jsonl"
    checkpoint_path = tmp_path / "last.pt"
    log_file = tmp_path / "train.log"
    final_model_dir = tmp_path / "final_model"

    write_jsonl(
        distilled_path,
        [make_distilled_record(max_len=max_len, vocab_size=256, used_rag=False)],
    )

    config = {
        "distilled_data_path": str(distilled_path),
        "max_len": max_len,
        "batch_size": 1,
        "lr": 1e-3,
        "checkpoint_path": str(checkpoint_path),
        "log_file": str(log_file),
        "final_model_dir": str(final_model_dir),
        "epochs": 1,
        "alpha": 0.7,
        "temperature": 2.0,
        "save_steps": 10,
    }

    student = FakeStudentModel(vocab_size=128)
    tokenizer = FakeTokenizer(vocab_size=256)
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="Teacher/student vocab mismatch"):
        train_student(config, student, tokenizer, device)
