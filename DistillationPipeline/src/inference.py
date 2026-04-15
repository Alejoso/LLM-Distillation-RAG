import torch

from src.data_processing import build_prompt


def ask_model(model, tokenizer, device, instruction, max_new_tokens=128):
    prompt = build_prompt(instruction)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Response:" in full_text:
        response = full_text.split("### Response:")[-1]
    else:
        response = full_text

    return response.strip()