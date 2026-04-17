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

CRITICAL RULE FOR ALL INSTRUCTIONS

Every instruction MUST be self-contained and specific.
A reader who has NOT read the document must be able to understand exactly
what is being asked WITHOUT needing to see the document.

THIS IS WRONG (too vague):
  "¿Qué fecha se menciona en el decreto?"
  "¿Cuál es el monto aprobado?"
  "¿Qué obligación establece el artículo 1?"
  "¿Qué rol cumple la entidad mencionada?"

THIS IS CORRECT (self-contained, specific):
  "¿En qué año fue firmado el Tratado Comercial entre Colombia y la Unión Soviética?"
  "¿Cuál fue el monto del crédito suplementario aprobado para cubrir la deuda nacional en el presupuesto 1881-1882?"
  "¿Qué obligaciones asumió la empresa soviética respecto al diseño de las represas del Alto Sinú?"
  "¿Bajo qué condición podía el Congreso colombiano aprobar el crédito de $80.000 según el artículo 1?"

The instruction MUST embed the specific names, amounts, dates, or topics
taken from the document. Do NOT use placeholders like [entidad] or [artículo X].
Replace them with the real value from the document.

QUESTION TYPE DEFINITIONS AND MANDATORY DISTRIBUTION

1. sobre_entidad_nombrada : 2 examples
   Ask about the role, responsibilities, or actions of a NAMED entity
   (a real person, institution, company, or country found in the document).
   The name must appear in the instruction itself.
   BAD:  "¿Qué hace la empresa mencionada en el convenio?"
   GOOD: "¿Qué responsabilidades asumió ENERGOPROEKT en la construcción de las represas del Alto Sinú?"

2. sobre_obligacion_o_prohibicion : 2 examples
   Ask about a specific obligation, right, or prohibition.
   The instruction must name the subject of the obligation AND the topic.
   BAD:  "¿Qué obliga el artículo 3?"
   GOOD: "¿Qué estaba obligado a hacer el gobierno colombiano respecto a las obras civiles de las represas según el convenio?"

3. sobre_condicion_o_requisito : 1 example
   Ask about a specific condition or requirement.
   The instruction must state what action or situation the condition governs.
   BAD:  "¿Qué condición se menciona?"
   GOOD: "¿Qué condición debía cumplirse para que el crédito suplementario de $80.000 fuera incluido en el presupuesto 1881-1882?"

4. sobre_dato_numerico_o_temporal : 2 examples
   Ask about a specific number, amount, date, or period.
   The instruction must name the concept the number refers to.
   BAD:  "¿Qué monto se menciona en el decreto?"
   GOOD: "¿Por cuántos años era válido el acuerdo de cooperación entre Colombia y la Unión Soviética firmado en 1968?"

5. sobre_alcance_o_aplicacion : 1 example
   Ask about who or what a specific provision applies to.
   The instruction must name the provision or topic.
   BAD:  "¿A quiénes aplica la ley?"
   GOOD: "¿A qué departamentos colombianos aplicaba la modificación de fronteras establecida en la ley de creación del Departamento de Nariño?"

6. sobre_relacion_entre_partes : 1 example
   Ask about the legal or contractual relationship between two NAMED parties.
   Both party names must appear in the instruction.
   BAD:  "¿Cuál es la relación entre las partes?"
   GOOD: "¿Cuál era la relación contractual entre el gobierno colombiano y ENERGOPROEKT según el convenio de 1968?"

7. sobre_consecuencia_o_efecto : 1 example
   Ask about the legal consequence or effect of a named provision or event.
   The instruction must name the specific provision or event.
   BAD:  "¿Qué consecuencia tiene el incumplimiento?"
   GOOD: "¿Qué efecto legal tuvo la promulgación de la ley del 6 de agosto de 1904 sobre el territorio del antiguo Departamento del Cauca?"

NOTE:
Always produce exactly 10 examples.
Every instruction must contain real names, amounts, dates, or topics from the document.
NEVER use generic placeholders in the final instruction.

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

Every instruction must be self-contained: a person who has not read the document
must understand exactly what is being asked just by reading the instruction.
This means the instruction must embed the real names, amounts, dates, and topics
from the document — not generic references like "el decreto" or "la entidad".

RULE — When referencing a legal document in the instruction:
If the instruction mentions a decreto, ley, convenio, tratado, or acuerdo,
it MUST identify it specifically using whatever is available in the document:
its number, its date, its title, or the parties involved.
BAD:  "¿Qué establece el decreto sobre la deuda?"
GOOD: "¿Qué establece el Decreto del 14 de septiembre de 1882 sobre el crédito suplementario para la deuda nacional?"
BAD:  "¿Qué dice el convenio entre las partes?"
GOOD: "¿Qué obligaciones estableció el Convenio Colombia-URSS de 1968 para la empresa ENERGOPROEKT?"

ABSOLUTELY FORBIDDEN instruction patterns:
- "Summarize the document..."
- "Identify the main idea..."
- "List key concepts..."
- "Reduce to X keywords..."
- "Classify content into essential and non-essential..."
- "What date is mentioned in the document?"
- "What amount is approved in article X?" (article number alone is not enough)
- "What does the entity mentioned do?" (must name the entity)
- Any instruction that uses "el documento", "el decreto", "la ley", "el artículo X",
  "la entidad", or "las partes" without naming the specific real-world referent

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
        if len(ex["response"]) < 3:
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
