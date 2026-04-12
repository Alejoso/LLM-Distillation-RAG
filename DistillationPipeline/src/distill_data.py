import json

import torch

from src.utils import load_json


def generate_distillation_jsonl(config, teacher, teacher_tokenizer, device):
    data = load_json(config["processed_path"])
    max_len = config["max_len"]

    with open(config["distilled_data_path"], "w", encoding="utf-8") as out:
        for sample in data:
            full_input = sample["prompt"] + sample["response"]

            inputs = teacher_tokenizer(
                full_input,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_len
            ).to(device)

            with torch.no_grad():
                outputs = teacher(**inputs)

            logits = torch.clamp(outputs.logits.squeeze(0), -10, 10).cpu().tolist()

            record = {
                "id": sample["id"],
                "prompt": sample["prompt"],
                "teacher_output": sample["response"],
                "logits": logits
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Distillation JSONL generated at: {config['distilled_data_path']}")