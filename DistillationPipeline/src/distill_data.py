import json
from typing import List, Tuple

import torch

from .data_processing import build_prompt
from .rag_adapter import retrieve_contexts
from .utils import load_json


def extract_response(full_text: str) -> str:
    if "### Response:" in full_text:
        return full_text.split("### Response:")[-1].strip()
    return full_text.strip()


def build_rag_prompt(instruction: str, retrieved_contexts: List[str]) -> str:
    context_block = "\n\n---\n\n".join(
        ctx.strip() for ctx in retrieved_contexts if ctx and ctx.strip()
    )

    return f"""### Instruction:
{instruction}

### Retrieved Context:
{context_block}

### Response:
"""


def prepare_teacher_prompt(sample: dict, config) -> Tuple[str, List[str]]:
    instruction = (sample.get("instruction") or "").strip()

    if not instruction:
        raise ValueError(f"Sample sin instruction válida: {sample}")

    retrieved_contexts: List[str] = []

    if config.get("use_rag_for_teacher", False):
        try:
            retrieved_contexts = retrieve_contexts(instruction, config)
        except Exception as e:
            print(f"[WARN] RAG retrieval failed for instruction: {instruction[:80]} | Error: {e}")
            retrieved_contexts = []

    if retrieved_contexts:
        prompt = build_rag_prompt(instruction, retrieved_contexts)
    else:
        prompt = build_prompt(instruction)

    return prompt, retrieved_contexts


def generate_distillation_jsonl(config, teacher, teacher_tokenizer, device):
    data = load_json(config["processed_path"])
    max_len = config["max_len"]
    max_new_tokens = config.get("max_new_tokens", 128)

    with open(config["distilled_data_path"], "w", encoding="utf-8") as out:
        for sample in data:
            prompt, retrieved_contexts = prepare_teacher_prompt(sample, config)

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

            if not teacher_output:
                teacher_output = "[EMPTY_RESPONSE]"

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
                "id": sample.get("id", -1),
                "instruction": (sample.get("instruction") or "").strip(),
                "prompt": prompt,
                "used_rag": bool(retrieved_contexts),
                "retrieved_contexts": retrieved_contexts,
                "teacher_output": teacher_output,
                "logits": logits
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Distillation JSONL generated at: {config['distilled_data_path']}")