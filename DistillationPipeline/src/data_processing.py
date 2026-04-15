from pathlib import Path

from .utils import load_json, save_json


def build_prompt(instruction: str, context: str = "") -> str:
    return f"""### Instruction:
{instruction}

### Context:
{context}

### Response:
"""


def create_sample_raw_data(path):
    sample_data = [
        {
            "instruction": "Explain what machine learning is.",
            "context": ""
        },
        {
            "instruction": "What is overfitting?",
            "context": ""
        },
        {
            "instruction": "Define neural networks.",
            "context": ""
        },
        {
            "instruction": "Explain supervised learning.",
            "context": ""
        },
        {
            "instruction": "What is a dataset?",
            "context": ""
        }
    ]

    save_json(sample_data, path)
    print(f"Sample raw dataset created at: {path}")


def load_raw_dataset(path):
    data = load_json(path)

    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    if not isinstance(data, list):
        raise ValueError("Raw dataset must be a list, or a dict with key 'data'.")

    return data


def process_dataset(config):
    raw_path = Path(config["raw_data_path"])
    if not raw_path.exists():
        create_sample_raw_data(raw_path)

    data = load_raw_dataset(raw_path)
    processed = []

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Invalid item at index {i}: {item}")
            continue

        if "instruction" not in item:
            print(f"Missing 'instruction' in item {i}: {item}")
            continue

        instruction = item["instruction"]
        context = item.get("context", "")

        processed.append({
            "id": i,
            "instruction": instruction,
            "context": context,
            "prompt": build_prompt(instruction, context)
        })

    save_json(processed, config["processed_path"])

    print(f"Processed dataset saved to: {config['processed_path']}")
    print(f"Total valid samples: {len(processed)}")

    if processed:
        print("\nProcessed example:")
        print(processed[0])

    return processed