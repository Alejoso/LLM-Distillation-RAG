# Legal Documentation Pipeline: A Research Study

## Abstract

This repository documents a complete research project journey: from discovering Colombia's legal database (SUIN) and manually extracting PDFs, to scaling up with automated web scraping, building intelligent preprocessing pipelines, and developing quality assessment metrics. The result is a clean, structured dataset of thousands of legal documents ready for computational analysis and Retrieval-Augmented Generation (RAG) applications.

---

## Table of Contents

1. [Research Pipeline Development](#research-pipeline-development)
   - [Phase 1: Discovery & Manual Foundation](#phase-1-discovery--manual-foundation)
   - [Phase 2 – Discovering the robots.txt and Sitemaps](#phase-2--discovering-the-robotstxt-and-sitemaps)
   - [Phase 3: Automated Web Scraping](#phase-3-automated-web-scraping)
   - [Phase 4 – Cleaning and Structuring the HTML Files](#phase-4--cleaning-and-structuring-the-html-files)
   - [Phase 5: Quality Assessment](#phase-5-quality-assessment)
   - [Phase 6: Domain MLM Preparation and Training Handoff](#phase-6-domain-mlm-preparation-and-training-handoff)
   - [Phase 7: LLM-Based Distillation Dataset Generation](#phase-7-llm-based-distillation-dataset-generation)
2. [Project Structure](#project-structure)
3. [Structure of a Cleaned File](#structure-of-a-cleaned-file)
   - [Metadata Header](#1-metadata-header)
   - [Main Legal Content](#2-main-legal-content)
4. [Methodology of Quality Assessment](#methodology-of-quality-assessment)
   - [Evaluation Metrics](#evaluation-metrics)
   - [Score Composition](#score-composition)
   - [Quality Classification](#quality-classification)
5. [Requirements and Installation](#requirements-and-installation)
   - [Requirements](#requirements)
   - [Installation](#installation)
6. [Pipeline Workflow](#pipeline-workflow)
7. [Usage Guides](#usage-guides)
   - [Main execution](#main-execution)
	 - [Step 1: Web Scraping - Download HTML Documents](#step-1-web-scraping---download-html-documents)
	 - [Step 2: HTML Processing - Extract and Clean Text](#step-2-html-processing---extract-and-clean-text)
   - [Other Scripts](#other-scripts)
	 - [PDF Processing - Extract and Clean PDF Documents](#pdf-processing---extract-and-clean-pdf-documents)
    - [Training Dataset Generation - Domain Sentence Corpus](#training-dataset-generation---domain-sentence-corpus)
    - [LLM-Based Distillation Dataset Generation](#llm-based-distillation-dataset-generation---instruction-response-pairs)
	 - [Quality Metrics Assessment - Standalone Evaluation](#quality-metrics-assessment---standalone-evaluation)

---

## Research Pipeline Development

### Phase 1: Discovery & Manual Foundation

I needed a dataset of Colombian legal documents for NLP research and discovered SUIN (Sistema Unica de Informacion Normativa) at https://www.suin-juriscol.gov.co/, the official repository of laws, decrees, and regulations. Since the platform does not offer bulk downloads, I manually downloaded approximately 10-20 PDF files and created my first script in `Scripts/ProcessPDFs/prepocessPDFs.py` which would interpret the PDF text and clean up noise or information that was not useful. This initial approach successfully validated the concept, but revealed a critical limitation: how was it going to automatically extract more than 11,000 PDFs and then clean them with the script to build the dataset?

### Phase 2 – Discovering the robots.txt and Sitemaps

As the number of manually downloaded PDFs increased, it became clear that continuing that way was not sustainable. I started exploring the SUIN website more carefully, navigating through different sections and observing how documents were linked. I wanted to understand whether there was an underlying structure that could help scale the extraction process.

During this exploration, I remembered that most public websites include a `robots.txt` file. Out of curiosity, I checked whether SUIN had one by manually typing:
https://www.suin-juriscol.gov.co/robots.txt

Inside that file, I found references to XML sitemap files. I was not initially searching for sitemaps specifically—I was simply trying to understand how the site exposed its structure. Opening those sitemap files revealed structured lists of document URLs maintained by the platform itself.

At that point, the approach began to shift. Instead of attempting to navigate the website as a user would, I realized I could rely on the sitemap to systematically identify available documents.

### Phase 3: Automated Web Scraping

Once I had access to the sitemap URLs, the next step was to automate the download of the documents listed there. For this purpose, I developed the script:
`Scripts/ProcessHTMLs/webScrappingData.py`

The script reads the sitemap files, extracts the document URLs, and iterates over them to download the corresponding HTML pages.

While implementing it, I incorporated several practical considerations:

- A 0.5-second delay between requests to avoid overwhelming the server.
- A retry mechanism with exponential backoff to handle temporary network failures.
- Duplicate detection to prevent re-downloading files that already exist.

Using this script, I downloaded more than 11.000 HTML documents in their raw form and stored them in `data/Laws/`.

At this stage, the extraction process was no longer manual. It became a repeatable procedure: the script can be executed again to retrieve new documents added to the sitemap without duplicating previously downloaded files.


### Phase 4 – Cleaning and Structuring the HTML Files

The downloaded HTML files contained much more than the legal text itself. Alongside the normative content, they included navigation menus, scripts, metadata, interface elements, and other boilerplate components that were not relevant for textual analysis.

To solve this, I developed the script:
`Scripts/ProcessHTMLs/preprocessHTMLs.py`

This script parses each HTML file, identifies the main content section, and removes unnecessary elements. It strips markup, eliminates interface components, normalizes spacing and line breaks, and converts the result into UTF-8 encoded plain text.

The output of this phase consists of more than 11.000 TXT files stored in `dataCleaned/Laws/`.

Although the extraction was automated, the resulting documents did not all have the same level of cleanliness. In some cases, short fragmented lines remained; in others, words were split across line breaks or certain structural markers were partially lost. This variability made it necessary to evaluate the quality of the cleaned files in a more systematic way, leading to the next phase of the project.

### Phase 5: Quality Assessment

As the cleaned documents showed noticeable variability in formatting and structure, I developed the script:
`Scripts/CleanlinessMetrics/compute_metrics.py`

The purpose of this script is to automatically evaluate and classify each document based on simple structural indicators of extraction quality.

The evaluation relies on three metrics:

- **Line Ratio (0–45 pts):** Measures the proportion of very short lines (≤3 characters). A high presence of short lines often indicates fragmented extraction.
- **Fragmentation Ratio (0–45 pts):** Detects words split across line breaks or separated by irregular spacing, suggesting formatting artifacts.
- **Header Integrity (0–10 pts):** Checks whether key legal structural markers (e.g., articles, sections, titles, clauses) are preserved.

Each document receives a composite score between 0 and 100 and is automatically classified into one of four categories:

- **HIGH (85–100):** Suitable for research use.
- **MEDIUM (70–84):** Usable but may require minor review.
- **LOW (50–69):** Requires manual verification or unusable.
- **DEFECTIVE (<50):** Not recommended for analysis.

After running this evaluation on the full set of 11.347 documents, 11021 were classified as HIGH or MEDIUM quality and retained in `dataCleaned/Laws/`. The other 326 documents classified as DEFECTIVE were moved to `dataCleaned/unusable_files/`.

### Phase 6: Domain MLM Preparation and Training Handoff

Once the cleaned dataset was ready, the next step was coordination with **Simon**, who is responsible for model training and distillation.

At this stage, I prepared a domain-focused training corpus so the MLM process could start with legal language that is explicitly in-domain.

The work completed in this phase included:

- Building a sentence-level dataset from `dataCleaned/Laws/` using `Scripts/ProccessDataTrainning/GenerateData.py`.
- Applying pattern-based cleanup (`CleaningPatterns.py`) plus text normalization to reduce extraction noise.
- Segmenting legal text with spaCy (`es_core_news_sm`) and filtering sentences by length to keep high-signal training samples.
- Exporting the resulting corpus as JSON (e.g., `datasetTrain.json`) as a direct handoff artifact for training.

With this handoff, Simon could begin domain-adaptive MLM pretraining and continue with the distillation workflow on a corpus aligned with Colombian legal language.

### Phase 7: LLM-Based Distillation Dataset Generation

Beyond the sentence-level corpus prepared in Phase 6, a second dataset generation approach was developed to produce structured instruction-response pairs using a large language model.

The notebook `Scripts/ProccessDataTrainning/GeneratingDataWithLLM.ipynb` uses **Llama-2-7b-chat** (via Hugging Face Transformers) to read each cleaned legal document and generate exactly 10 question-answer examples designed for information distillation training.

Each generated example follows the format:
- **Instruction:** A question or task about the legal document.
- **Context:** Always empty (the model must rely on internalized knowledge from the prompt).
- **Response:** A compressed, synthesized answer derived strictly from the document content.

The 10 examples per document follow a mandatory distribution across 9 question types:

| Type | Count | Description |
|------|-------|-------------|
| `idea_central` | 2 | Capture the core idea in a single sentence |
| `resumen_3_niveles` | 1 | Summary at 3 levels: 1 sentence, 3 sentences, and 5 bullet points |
| `esencial_vs_accesorio` | 1 | Classify content into essential and non-essential |
| `estructura_logica` | 1 | Describe the logical structure of the document |
| `reescritura_simplificada` | 1 | Plain-language rewrite accessible to a layperson |
| `intencion_autor` | 1 | Identify the purpose of the document |
| `conceptos_clave` | 1 | List key concepts with one-line definitions |
| `reduccion_extrema` | 1 | Select exactly 5 keywords that capture the essence |
| `conexiones_internas` | 1 | Explain how the ideas in the document relate to each other |

The pipeline includes:
- **Prompt engineering** with strict formatting rules so that the output can be parsed programmatically using regex.
- **Validation** that filters out examples with instructions shorter than 10 characters or responses shorter than 30 characters.
- **Document truncation** to 4,000 characters to fit within the model's context window.
- **Output** saved as a JSON file (`datasetTrain.json`) with the same structure used in Phase 6.

This approach complements the sentence-level corpus by producing higher-level reasoning and comprehension examples that can be used for instruction-tuning or knowledge distillation.

## Project Structure

## Structure of a Cleaned File

Each cleaned `.txt` file follows a standardized structure composed of two clearly separated sections:

1. Metadata Header  
2. Main Legal Content  

The metadata block always appears at the top of the file, followed immediately by the cleaned legal text.

### 1. Metadata Header

The file begins with a structured metadata block using a strict `KEY: VALUE` format.

All metadata fields are written in uppercase.

**Fields included:**

- `TIPO`
- `NUMERO`
- `ANIO`
- `ESTADO`
- `ENTIDAD`
- `SUBTIPO`
- `FECHA_EXPEDICION`
- `FECHA_PUBLICACION`
- `FUENTE`
- `QUALITY_SCORE`
- `QUALITY_STATUS`

**Example:**

```text
TIPO: LEY
NUMERO: 1
ANIO: 1904
ESTADO: Vigencia en Estudio
ENTIDAD: CONGRESO DE LA REPUBLICA
SUBTIPO: LEY ORDINARIA
FECHA_EXPEDICION:
FECHA_PUBLICACION: 13/08/1904
FUENTE: DIARIO OFICIAL. AÑO XL. N. 12145. 13, AGOSTO, 1904. PÁG. 1.
QUALITY_SCORE: 100
QUALITY_STATUS: HIGH
```

### 2. Main Legal Content

After the metadata section, the cleaned legal content begins immediately.

This section contains the normalized and structured legal text extracted from the original document.

It may include:

- Articles (`ARTÍCULO`)
- Sections
- Paragraphs
- Clauses

**Example:**

```text
ARTÍCULO 1. Objeto de la ley...
ARTÍCULO 2. Definiciones...
ARTÍCULO 3. Disposiciones finales...
```

## Methodology of Quality Assessment

The quality assessment module evaluates each cleaned legal document using a quantitative scoring system designed to measure structural integrity and formatting reliability.

Each document receives a composite score between **0 and 100**, based on three weighted structural metrics.

---

### Evaluation Metrics

The final score is computed using the following components:

- **Line Ratio (0–45 pts)**  
  - Measures the proportion of very short lines (≤ 3 characters).  
  - A high presence of short lines often indicates fragmented extraction, broken formatting, or OCR artifacts.  
  - Fewer short lines result in a higher score.

- **Fragmentation Ratio (0–45 pts)**  
  - Detects words split across line breaks or separated by irregular spacing.  
  - High fragmentation suggests formatting artifacts introduced during PDF or HTML extraction.  
  - Lower fragmentation increases the score.

- **Header Integrity (0–10 pts)**  
  Verifies whether key legal structural markers are preserved, such as:
  - `ARTÍCULO`
  - Section headings
  - Titles
  - Clauses  

  Documents that maintain their hierarchical legal structure receive the maximum score in this category.

---

### Score Composition

The composite quality score is calculated as a weighted sum:

- Line Structure: 45%
- Fragmentation Detection: 45%
- Structural Integrity: 10%


---

### Quality Classification

Based on the final `QUALITY_SCORE`, each document is automatically classified into one of four categories:

- **HIGH (85–100)**  
  Suitable for research use.  
  Clean formatting, minimal fragmentation, and preserved structural hierarchy.

- **MEDIUM (70–84)**  
  Usable but may require minor review.  
  Small formatting inconsistencies may be present.

- **LOW (50–69)**  
  Requires manual verification.  
  Noticeable structural or formatting issues.

- **DEFECTIVE (<50)**  
  Not recommended for analysis.  
  Severe formatting errors, high fragmentation, or missing structural markers.

Documents classified as LOW or DEFECTIVE may be moved to the `unusable_files`.

## Requirements and Installation
This section provides instructions for setting up the environment and installing necessary dependencies to run the scripts in this repository.

---

### Requirements

- Python 3.10 or higher
- pip (Python package manager)
- Virtual environment (recommended)

All required Python dependencies are listed in:
- `requirements.txt`


---

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/JuanTrujilloM/Legal-Docs-Pipeline
cd Legal-Documentation-Pipeline
```

2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```
4. **Run the scripts**

The usage of each script is explained in its respective section of the README. Make sure to follow the instructions for each phase of the pipeline to ensure proper execution.

## Pipeline Workflow

1. **Web Scraping**  
   Use `Scripts/ProcessHTMLs/webScrappingData.py` to download HTML files from the SUIN sitemaps.
2. **Preprocessing**
   Use `Scripts/ProcessHTMLs/preprocessHTMLs.py` to clean and structure the downloaded HTML files into plain text.
3. **Quality Assessment**
   Use `Scripts/CleanlinessMetrics/compute_metrics.py` to evaluate the quality of the cleaned text files and classify them based on the computed metrics.

## Usage Guides
### Main execution

#### Step 1: Web Scraping - Download HTML Documents

Execute the web scraping script to systematically download all legal documents from SUIN's sitemaps:

```bash
python3 Scripts/ProcessHTMLs/webScrappingData.py --output data/Laws
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sitemap` | `https://www.suin-juriscol.gov.co/sitemapleyes.xml` | URL of the sitemap to scrape |
| `--output` | (required) | Directory where HTML files will be saved |
| `--sleep` | 0.5 | Delay in seconds between requests |
| `--max-retries` | 6 | Maximum number of retry attempts for failed downloads |
| `--backoff` | 1.5 | Exponential backoff factor for retries |
| `--failed-log` | `failed_downloads.txt` | File to log failed download attempts |
---

#### Step 2: HTML Processing - Extract and Clean Text

Process the downloaded HTML files to extract clean legal text:

```bash
python3 Scripts/ProcessHTMLs/preprocessHTMLs.py --input data/Laws --output dataCleaned/Laws
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `--input` / `-i` | (required) | Directory containing HTML files to process |
| `--output` / `-o` | (required) | Directory where cleaned TXT files will be saved |


**Note:** This script integrates `Scripts/CleanlinessMetrics/compute_metrics.py` to automatically assess quality metrics (Line Ratio, Fragmentation Ratio, and Header Integrity) and classify documents. HIGH and MEDIUM quality documents are saved to `dataCleaned/Laws/`, while LOW and DEFECTIVE documents are moved to `dataCleaned/unusable_files/`.

---

### Other Scripts

#### PDF Processing - Extract and Clean PDF Documents

If you have PDF files instead of HTML files, you can use the standalone PDF processing script:

```bash
python3 Scripts/ProcessPDFs/processPDFs.py --input input.pdf --output output.txt
```

Or process an entire directory of PDFs:

```bash
python3 Scripts/ProcessPDFs/processPDFs.py --input data/PDFs --output dataCleaned/PDFs
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `--input` / `-i` | (required) | Single PDF file or directory with PDF files |
| `--output` / `-o` | (required) | Output file or directory for cleaned TXT files |
| `--extensions` / `-e` | (optional) | File extensions to process (default: .pdf) |

---

#### Training Dataset Generation - Domain Sentence Corpus

To generate a sentence-level JSON dataset for domain MLM training from cleaned legal TXT files, use:

```bash
python3 Scripts/ProccessDataTrainning/GenerateData.py --input dataCleaned/Laws --output datasetTrain.json
```

You can also run it with custom filtering limits:

```bash
python3 Scripts/ProccessDataTrainning/GenerateData.py --input dataCleaned/Laws --output datasetTrain.json --min-chars 100 --max-chars 200 --max-sentences 5000
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `--input` / `-i` | (optional) | Directory containing cleaned TXT files (default: `dataCleaned/Laws`) |
| `--output` / `-o` | (optional) | Output JSON file path (default: `dataset_leyes.json`) |
| `--min-chars` | (optional) | Minimum sentence length in characters (default: `100`) |
| `--max-chars` | (optional) | Maximum sentence length in characters (default: `200`) |
| `--max-sentences` | (optional) | Maximum number of sentences to export (default: `5000`) |

---

#### LLM-Based Distillation Dataset Generation - Instruction-Response Pairs

To generate instruction-response training pairs from cleaned legal documents using Llama-2-7b-chat, open and run the notebook:

```
Scripts/ProccessDataTrainning/GeneratingDataWithLLM.ipynb
```

**Prerequisites:**

- A GPU-enabled environment (e.g., Google Colab with CUDA)
- A Hugging Face account with access to `meta-llama/Llama-2-7b-chat-hf`
- Authentication via `huggingface_hub login()`

**Configuration (editable in the notebook):**

| Variable | Default | Description |
|----------|---------|-------------|
| `LAWS_FOLDER` | `dataCleaned/Laws` | Directory containing cleaned TXT files |
| `OUTPUT_FILE` | `dataCleaned/datasetTrain.json` | Output JSON file path |
| `max_files` | `2` | Maximum number of documents to process |
| `temperature` | `0.2` | Sampling temperature for generation |
| `max_new_tokens` | `3000` | Maximum tokens generated per document |

**Output format:**

```json
{
  "data": [
    {
      "instruction": "¿Cuál es la idea central de esta ley?",
      "context": "",
      "response": "La ley establece..."
    }
  ]
}
```

**Note:** Only documents that produce exactly 10 valid examples are included in the final dataset. Documents that fail parsing or validation are skipped with a diagnostic message.

---

#### Quality Metrics Assessment - Standalone Evaluation

You can also run the quality assessment script independently on any cleaned text file:

```bash
python3 Scripts/CleanlinessMetrics/compute_metrics.py dataCleaned/Laws/document.txt
```