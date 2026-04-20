import json
import types
from pathlib import Path

import torch
import torch.nn as nn


class FakeBatch(dict):
    def to(self, device):
        for k, v in self.items():
            if torch.is_tensor(v):
                self[k] = v.to(device)
        return self


class FakeTokenizer:
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def _encode_text(self, text: str):
        ids = []
        for ch in text:
            code = ord(ch)
            if code >= self.vocab_size:
                code = 1
            if code == 0:
                code = 1
            ids.append(code)
        return ids

    def __call__(
        self,
        text,
        return_tensors="pt",
        padding=None,
        truncation=False,
        max_length=None,
    ):
        ids = self._encode_text(text)

        if truncation and max_length is not None:
            ids = ids[:max_length]

        attention_mask = [1] * len(ids)

        if padding == "max_length" and max_length is not None:
            pad_len = max_length - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len

        return FakeBatch(
            {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }
        )

    def decode(self, ids, skip_special_tokens=True):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        chars = []
        for token_id in ids:
            if token_id == self.pad_token_id:
                continue
            if 0 < token_id < 256:
                chars.append(chr(token_id))
        return "".join(chars)

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("fake-tokenizer", encoding="utf-8")


class FakeTeacherModel(nn.Module):
    def __init__(self, tokenizer, vocab_size=256, generated_text="### Response:\nTeacher answer"):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.generated_text = generated_text

    def generate(self, input_ids, attention_mask=None, max_new_tokens=None, do_sample=None, pad_token_id=None):
        return self.tokenizer(self.generated_text, return_tensors="pt")["input_ids"]

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        logits = torch.linspace(-5, 5, steps=self.vocab_size, dtype=torch.float32)
        logits = logits.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1)
        return types.SimpleNamespace(logits=logits)


class ExtremeTeacherModel(FakeTeacherModel):
    def __init__(self, tokenizer, vocab_size=256, generated_text="   "):
        super().__init__(tokenizer=tokenizer, vocab_size=vocab_size, generated_text=generated_text)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        base = torch.linspace(-20, 20, steps=self.vocab_size, dtype=torch.float32)
        logits = base.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1)
        return types.SimpleNamespace(logits=logits)


class FakeStudentModel(nn.Module):
    def __init__(self, vocab_size=256, hidden_size=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = types.SimpleNamespace(vocab_size=vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        logits = self.lm_head(x)
        return types.SimpleNamespace(logits=logits)

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "pytorch_model.bin")


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_distilled_record(max_len=32, vocab_size=256, used_rag=False):
    prompt = "### Instruction:\nHi\n\n### Response:\n"
    teacher_output = "Hello"
    logits = [[0.1] * vocab_size for _ in range(max_len)]
    return {
        "id": 0,
        "instruction": "Hi",
        "prompt": prompt,
        "used_rag": used_rag,
        "retrieved_contexts": ["ctx-1"] if used_rag else [],
        "teacher_output": teacher_output,
        "logits": logits,
    }
