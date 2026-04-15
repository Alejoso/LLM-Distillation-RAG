import json

import torch
from torch.utils.data import Dataset


class DistillationDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        with open(path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        prompt = sample["prompt"]
        teacher_output = sample["teacher_output"]
        full_text = prompt + teacher_output

        tokens = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        )

        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        teacher_logits = torch.tensor(sample["logits"], dtype=torch.float32)

        if teacher_logits.size(0) > self.max_len:
            teacher_logits = teacher_logits[:self.max_len]
        elif teacher_logits.size(0) < self.max_len:
            pad_shape = (self.max_len - teacher_logits.size(0), teacher_logits.size(1))
            padding = torch.zeros(pad_shape, dtype=teacher_logits.dtype)
            teacher_logits = torch.cat([teacher_logits, padding], dim=0)

        response_mask = torch.zeros(self.max_len, dtype=torch.float32)
        prompt_len = min(prompt_tokens["input_ids"].size(1), self.max_len)
        pad_token_id = self.tokenizer.pad_token_id

        for j in range(prompt_len, self.max_len):
            if input_ids[j].item() != pad_token_id:
                response_mask[j] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "teacher_logits": teacher_logits,
            "response_mask": response_mask,
            "used_rag": sample.get("used_rag", False),
        }