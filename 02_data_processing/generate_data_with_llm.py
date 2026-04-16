"""
GeneratingDataWithLLM.py

Generates a training dataset from Colombian legal documents using
meta-llama/Llama-2-7b-chat-hf. Designed to run on HPC environments (Apolo).

Usage:
    python GeneratingDataWithLLM.py \
        --laws_folder /path/to/dataCleaned/Laws \
        --output_file /path/to/dataCleaned/datasetTrain.json \
        --hf_token <your_token> \
        [--max_files N]
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Model setup
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"


def load_model(hf_token: str):
    from huggingface_hub import login

    login(token=hf_token)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    logger.info("Loading model (float16, device_map=auto)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )

    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("Model device: %s", model.device)

    return model, tokenizer

# Generation
def generate_questions(text: str, model, tokenizer) -> str:
    prompt = f"""<s>[INST] <<SYS>>
You are a legal analyst specialized in Colombian law.

You generate training data for information distillation from Colombian legal documents.

You MUST follow all rules strictly. You are NOT allowed to invent information.
You MUST ONLY use the content of the provided document.
<</SYS>>

TASK

Read the Colombian legal document and generate EXACTLY TEN question-answer examples
designed to train a model in information distillation (extracting, compressing, and
reorganizing the essential content of a text).

Each example MUST include:
- instruction
- context
- response

The "context" field MUST ALWAYS be empty.

QUESTION TYPE DEFINITIONS AND MANDATORY DISTRIBUTION

1. idea_central : 2 examples
   The response must capture the core idea in a single sentence.

2. resumen_3_niveles : 1 example
   The response must provide:
   - Summary in 1 sentence
   - Summary in 3 sentences
   - 5 key bullet points

3. esencial_vs_accesorio : 1 example
   The response must classify content into essential and non-essential.

4. estructura_logica : 1 example
   The response must describe the logical structure of the document.

5. reescritura_simplificada : 1 example
   The response must be a plain-language rewrite accessible to a layperson.

6. intencion_autor : 1 example
   The response must identify the purpose of the document.

7. conceptos_clave : 1 example
   The response must list key concepts with one-line definitions.

8. reduccion_extrema : 1 example
   The response must select exactly 5 keywords that capture the essence.

9. conexiones_internas : 1 example
   The response must explain how the ideas relate to each other.

NOTE:
If the document is too short for resumen_3_niveles, replace it with
another idea_central.
Always produce exactly 10 examples.

STRICT RULES

1. Use ONLY information explicitly written in the document.
2. DO NOT invent numbers, facts, or details.
3. DO NOT assume missing information.
4. DO NOT use external knowledge.
5. If information is not present, DO NOT create it.
6. All questions and responses MUST be in Spanish.
7. Adapt the wording naturally — do not repeat the exact same question every time.

OUTPUT FORMAT (VERY STRICT)

You MUST return ONLY plain text using the following structure.
Do NOT return JSON.
Do NOT include explanations.
Do NOT include comments.
Do NOT include notes.

You MUST generate different types of instructions.
Do NOT repeat instruction types.
Use a mix of:
- definition
- extraction
- classification
- reasoning
- legal interpretation
- obligations
- rights
- entities
- dates
- conditions
- prohibitions
- scope of law

You MUST extract information ONLY from the document.
You are NOT allowed to infer, assume, or add information.

IMPORTANT:
Do NOT copy long parts of the document.
Responses must be compressed, synthesized, and rewritten.
Do NOT quote the document unless strictly necessary.
Each response must be shorter than the original document.

FORMAT:

You MUST follow the format EXACTLY.
If you do not follow the format, your answer is WRONG.

Do NOT write introductions.
Do NOT write explanations.
Do NOT write any text before "Ejemplo 1".
Do NOT write any text after "Ejemplo 10".

You MUST write EXACTLY this format:

Ejemplo 1:
Instruction: ...
Context:
Response: ...

Ejemplo 2:
Instruction: ...
Context:
Response: ...

Repeat until Ejemplo 10.

Your answer will be parsed by a computer program using regex.
If you change the format, the program will fail.

VALIDATION (INTERNAL)

Before answering:
- Verify all answers come from the document
- Verify exactly 10 examples are generated
- Verify the format is respected
- Verify context is always empty

LEGAL DOCUMENT

{text}

Generate the examples now.

[/INST]"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=3000,
        temperature=0.2,
        do_sample=True,
        top_p=0.3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = output[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)

    logger.info("\n================ RAW MODEL OUTPUT ================\n%s\n==================================================\n", result)

    return result

# Parsing & validation
def parse_examples(text: str) -> dict:
    # Split by "Ejemplo N:" and parse each block independently.
    # A single big regex with DOTALL causes the response group to consume
    # multiple examples at once when responses are multiline.
    blocks = re.split(r"Ejemplo\s*\d+:\s*\n", text)
    blocks = [b for b in blocks if b.strip()]

    data = []
    for block in blocks:
        inst_match = re.search(
            r"(?:Instrucción|Instruction):\s*(.*?)(?:\n|$)", block, re.IGNORECASE
        )
        resp_match = re.search(
            r"(?:Respuesta|Response):\s*([\s\S]*?)$", block, re.IGNORECASE | re.DOTALL
        )

        if inst_match and resp_match:
            data.append({
                "instruction": inst_match.group(1).strip(),
                "context": "",
                "response": resp_match.group(1).strip(),
            })

    return {"data": data}


def validate_examples(parsed: dict) -> dict:
    valid_data = []

    for ex in parsed["data"]:
        if len(ex["instruction"]) < 10:
            continue
        if len(ex["response"]) < 30:
            continue
        if ex["context"] != "":
            continue

        valid_data.append(ex)

    return {"data": valid_data}

# Main processing loop
def process_documents(laws_folder: Path, output_file: Path, model, tokenizer, max_files: int):
    dataset = {"data": []}
    processed_count = 0

    txt_files = [f for f in os.listdir(laws_folder) if f.endswith(".txt")]
    logger.info("Found %d .txt files in %s", len(txt_files), laws_folder)

    for file in txt_files:
        if processed_count >= max_files:
            logger.info("Reached limit of %d files.", max_files)
            break

        path = laws_folder / file
        logger.info("Processing %s (%d/%d)", file, processed_count + 1, max_files)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        if "CONTENIDO:" in text:
            text = text.split("CONTENIDO:", 1)[1].strip()

        text = text[:4000]

        try:
            output = generate_questions(text, model, tokenizer)

            parsed = parse_examples(output)
            parsed = validate_examples(parsed)

            logger.info("Examples detected: %d", len(parsed["data"]))

            for i, ex in enumerate(parsed["data"]):
                logger.info(
                    "--- Example %d ---\nInstruction: %s\nResponse: %s",
                    i + 1,
                    ex["instruction"][:100],
                    ex["response"][:120],
                )

            if len(parsed["data"]) == 10:
                dataset["data"].extend(parsed["data"])
                processed_count += 1
                logger.info("Successfully processed %s", file)
            else:
                logger.warning(
                    "Expected 10 examples, got %d for %s. First 500 chars:\n%s",
                    len(parsed["data"]),
                    file,
                    output[:500],
                )

        except Exception as e:
            logger.error("Unexpected error for %s: %s", file, e)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logger.info("Dataset saved: %s", output_file)
    logger.info("Total samples: %d", len(dataset["data"]))

# Entry point
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training dataset from Colombian legal documents using Llama-2."
    )
    parser.add_argument(
        "--laws_folder",
        type=Path,
        required=True,
        help="Path to the folder containing cleaned .txt legal documents.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path where the output datasetTrain.json will be saved.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="Hugging Face access token (needed for gated Llama-2 model).",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process. Defaults to all files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.laws_folder.exists():
        logger.error("laws_folder does not exist: %s", args.laws_folder)
        sys.exit(1)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    total_files = len([f for f in os.listdir(args.laws_folder) if f.endswith(".txt")])
    max_files = args.max_files if args.max_files is not None else total_files

    model, tokenizer = load_model(args.hf_token)
    process_documents(args.laws_folder, args.output_file, model, tokenizer, max_files)
