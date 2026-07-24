"""
GeneratingDataWithLLM_ollama_test.py

Versión de prueba local usando Ollama en lugar de HuggingFace Transformers.
Mantiene el mismo prompt y lógica del script original (GeneratingDataWithLLM.py).

Requiere:
    - Ollama corriendo localmente: `ollama serve`
    - Modelo descargado, por defecto llama3: `ollama pull llama3`

Usage:
    python GeneratingDataWithLLM_ollama_test.py \
        --laws_folder /path/to/dataCleaned/Laws \
        --output_file /path/to/output/datasetTrain_test.json \
        [--model llama3] \
        [--max_files 1]
"""

import argparse
import json
import logging
import os
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"


# ---------------------------------------------------------------------------
# Generation via Ollama REST API (sin dependencias extra)
# ---------------------------------------------------------------------------

def generate_questions(text: str, model: str) -> str:
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

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.3,
            "num_predict": 3000,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            result = body.get("response", "")
    except urllib.error.URLError as e:
        logger.error("Could not reach Ollama at %s — is `ollama serve` running? (%s)", OLLAMA_URL, e)
        sys.exit(1)

    logger.info("\n================ RAW MODEL OUTPUT ================\n%s\n==================================================\n", result)

    return result


# ---------------------------------------------------------------------------
# Parsing & validation (idénticos al script original)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_documents(laws_folder: Path, output_file: Path, model: str, max_files: int):
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
            output = generate_questions(text, model)

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test local con Ollama — misma lógica que GeneratingDataWithLLM.py."
    )
    parser.add_argument(
        "--laws_folder",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataCleaned" / "Laws",
        help="Carpeta con los .txt de leyes. Default: ../../dataCleaned/Laws",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataCleaned" / "datasetTrain_test.json",
        help="Archivo JSON de salida. Default: ../../dataCleaned/datasetTrain_test.json",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="Nombre del modelo Ollama a usar. Default: llama3",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=1,
        help="Número máximo de archivos a procesar. Default: 1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.laws_folder.exists():
        logger.error("laws_folder no existe: %s", args.laws_folder)
        sys.exit(1)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    total_files = len([f for f in os.listdir(args.laws_folder) if f.endswith(".txt")])
    max_files = min(args.max_files, total_files)

    logger.info("Modelo Ollama: %s", args.model)
    logger.info("Archivos a procesar: %d", max_files)

    process_documents(args.laws_folder, args.output_file, args.model, max_files)
