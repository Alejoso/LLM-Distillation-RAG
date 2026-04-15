import json

import torch

from .utils import load_json


def extract_response(full_text: str) -> str:
    if "### Response:" in full_text:
        return full_text.split("### Response:")[-1].strip()
    return full_text.strip()


def generate_distillation_jsonl(config, teacher, teacher_tokenizer, device):
    data = load_json(config["processed_path"])
    max_len = config["max_len"]
    max_new_tokens = config.get("max_new_tokens", 128)

    with open(config["distilled_data_path"], "w", encoding="utf-8") as out:
        for sample in data:
            prompt = sample["prompt"]

            # 1. Teacher generates the answer from the prompt
            generation_inputs = teacher_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_len
            ).to(device)

            with torch.no_grad():
                generated_ids = teacher.generate(
                    **generation_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=teacher_tokenizer.pad_token_id
                )

            full_generated_text = teacher_tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )

            teacher_output = extract_response(full_generated_text)

            # Safety fallback
            if not teacher_output:
                teacher_output = "[EMPTY_RESPONSE]"

            # 2. Build full sequence for teacher forcing and logits extraction
            full_input = prompt + teacher_output

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
                "instruction": sample.get("instruction", ""),
                "context": sample.get("context", ""),
                "prompt": prompt,
                "teacher_output": teacher_output,
                "logits": logits
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Distillation JSONL generated at: {config['distilled_data_path']}")