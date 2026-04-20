import importlib
import sys
import types

import torch


def test_main_orchestrates_pipeline_steps_in_order(monkeypatch):
    calls = []

    fake_config = types.ModuleType("config")
    fake_config.CONFIG = {"seed": 42}
    fake_config.ensure_directories = lambda: calls.append("ensure_directories")

    fake_auth = types.ModuleType("src.auth")
    fake_auth.hf_login_if_needed = lambda: calls.append("hf_login_if_needed")

    fake_data_processing = types.ModuleType("src.data_processing")
    fake_data_processing.process_dataset = lambda config: calls.append("process_dataset")

    fake_distill_data = types.ModuleType("src.distill_data")
    fake_distill_data.generate_distillation_jsonl = (
        lambda config, teacher, teacher_tokenizer, device: calls.append("generate_distillation_jsonl")
    )

    fake_models = types.ModuleType("src.models")

    def fake_load_teacher_and_student(config):
        calls.append("load_teacher_and_student")
        teacher_tokenizer = object()
        student_tokenizer = object()
        teacher = object()
        student = object()
        device = torch.device("cpu")
        return teacher_tokenizer, student_tokenizer, teacher, student, device

    fake_models.load_teacher_and_student = fake_load_teacher_and_student

    fake_trainer = types.ModuleType("src.trainer")
    fake_trainer.train_student = lambda config, student, student_tokenizer, device: calls.append("train_student")

    fake_utils = types.ModuleType("src.utils")
    fake_utils.set_seed = lambda seed: calls.append("set_seed")

    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "src.auth", fake_auth)
    monkeypatch.setitem(sys.modules, "src.data_processing", fake_data_processing)
    monkeypatch.setitem(sys.modules, "src.distill_data", fake_distill_data)
    monkeypatch.setitem(sys.modules, "src.models", fake_models)
    monkeypatch.setitem(sys.modules, "src.trainer", fake_trainer)
    monkeypatch.setitem(sys.modules, "src.utils", fake_utils)

    sys.modules.pop("run_pipeline", None)
    run_pipeline = importlib.import_module("run_pipeline")
    run_pipeline.main()

    assert calls == [
        "ensure_directories",
        "set_seed",
        "hf_login_if_needed",
        "process_dataset",
        "load_teacher_and_student",
        "generate_distillation_jsonl",
        "train_student",
    ]

