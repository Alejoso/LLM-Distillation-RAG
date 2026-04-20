from src.dataset import DistillationDataset
from tests.helpers import FakeTokenizer, make_distilled_record, write_jsonl


def test_distillation_dataset_getitem_builds_training_tensors_and_mask(tmp_path):
    max_len = 24
    path = tmp_path / "train.jsonl"
    record = make_distilled_record(max_len=max_len, vocab_size=256, used_rag=True)
    write_jsonl(path, [record])

    tokenizer = FakeTokenizer()
    dataset = DistillationDataset(path=str(path), tokenizer=tokenizer, max_len=max_len)

    item = dataset[0]

    assert set(item.keys()) == {
        "input_ids",
        "attention_mask",
        "teacher_logits",
        "response_mask",
        "used_rag",
    }
    assert item["input_ids"].shape[0] == max_len
    assert item["attention_mask"].shape[0] == max_len
    assert item["teacher_logits"].shape == (max_len, 256)
    assert item["response_mask"].shape[0] == max_len
    assert item["response_mask"].sum().item() > 0
    assert item["used_rag"] is True
