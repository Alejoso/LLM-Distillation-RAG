# LLM Distillation with RAG for Colombian Legal Documents

A research pipeline for **knowledge distillation** of Large Language Models (LLMs) enhanced with **Retrieval-Augmented Generation (RAG)**, applied to the domain of Colombian legal documents.

## Overview

This project implements a complete pipeline that:

1. **Collects** legal documents from the Colombian SUIN portal via web scraping
2. **Processes** raw HTML into clean, structured text with quality scoring
3. **Indexes** documents in a vector database for retrieval (RAG)
4. **Generates** training data using a teacher LLM with RAG context
5. **Distills** knowledge from a large teacher model (Llama-2-7B) into a smaller student model (TinyLlama-1.1B)
6. **Evaluates** output quality using an LLM-as-Judge approach

## Project Structure

```
.
├── 01_data_collection/          # Web scraping & HTML preprocessing
│   ├── web_scraping.py          # SUIN portal scraper with retry logic
│   ├── preprocess_htmls.py      # HTML → structured TXT conversion
│   └── text_normalization.py    # 40+ regex rules for legal text cleanup
│
├── 02_data_processing/          # Data quality & training data generation
│   ├── compute_metrics.py       # Quality scoring (line ratio, fragmentation, headers)
│   ├── generate_stats.py        # Dataset statistics & visualizations
│   ├── generate_data_with_llm.py        # Training data generation via Llama-2
│   └── generate_data_ollama_test.py     # Local testing variant using Ollama
│
├── 03_rag/                      # Retrieval-Augmented Generation
│   ├── create_db/               # Chroma vector DB creation with BGE-M3 embeddings
│   │   ├── create_db.py
│   │   └── define_BGEM3_embeddings.py
│   └── query_db/                # Query interface with BGE reranker
│       ├── query.py
│       ├── reranker_BGE.py
│       └── test_cases.json      # 20 test cases (factual, inferential, multi-chunk)
│
├── 04_distillation/             # Knowledge distillation pipeline
│   ├── config.py                # Centralized configuration
│   ├── run_pipeline.py          # Main pipeline orchestrator
│   ├── chat.py                  # Interactive inference interface
│   ├── src/
│   │   ├── auth.py              # Hugging Face authentication
│   │   ├── data_processing.py   # Dataset loading & prompt building
│   │   ├── dataset.py           # PyTorch Dataset for distillation
│   │   ├── distill_data.py      # Teacher inference & logits extraction
│   │   ├── models.py            # Model loading (teacher & student)
│   │   ├── rag_adapter.py       # RAG integration for teacher prompts
│   │   ├── trainer.py           # Student training with KL+CE hybrid loss
│   │   ├── inference.py         # Model inference utilities
│   │   └── utils.py             # Seeds, I/O helpers
│   └── data/                    # Training data (raw, processed, distilled)
│
├── 05_evaluation/               # Quality evaluation
│   └── clarity_judge/           # LLM-as-Judge evaluation framework
│       ├── cli.py               # Command-line interface
│       ├── client.py            # Ollama API client
│       ├── evaluator.py         # Weighted scoring (accuracy, relevance, completeness, clarity)
│       ├── models.py            # Pydantic data models
│       ├── prompts.py           # Evaluation prompt templates
│       └── utils.py             # I/O and summary statistics
│
├── tests/                       # Unit tests
│   └── unit/
│       ├── test_preprocess_htmls.py
│       ├── test_quality_metrics.py
│       └── test_text_normalization.py
│
├── docs/                        # Documentation & research
│   ├── spikes/                  # Technical investigation documents
│   ├── TESTING.md               # Testing conventions
│   └── hello_world_distillation.py  # Original pedagogical example
│
├── requirements.txt             # Consolidated dependencies
└── LICENSE                      # MIT License
```

## Pipeline Flow

```
[SUIN Portal] ──scrape──► [Raw HTML]
                              │
                         preprocess
                              │
                              ▼
                      [Clean TXT + Quality Score]
                              │
                    ┌─────────┴─────────┐
                    │                   │
               index in DB        generate Q&A
                    │            (via teacher LLM)
                    ▼                   │
              [Chroma Vector DB]        │
                    │                   │
                    └─────┬─────────────┘
                          │
                    RAG-augmented
                    teacher inference
                          │
                          ▼
                  [Distillation Data]
                   (prompts + logits)
                          │
                     train student
                   (KL div + CE loss)
                          │
                          ▼
                  [TinyLlama Fine-tuned]
                          │
                     evaluate with
                     LLM-as-Judge
                          │
                          ▼
                  [Quality Metrics]
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### 1. Data Collection

```bash
# Scrape legal documents
python 01_data_collection/web_scraping.py

# Process HTML to clean text
python 01_data_collection/preprocess_htmls.py -i data/raw_html -o data/cleaned
```

### 2. Create Vector Database

```bash
python 03_rag/create_db/create_db.py
```

### 3. Generate Training Data

```bash
python 02_data_processing/generate_data_with_llm.py
```

### 4. Run Distillation Pipeline

```bash
export HF_TOKEN=your_huggingface_token
python 04_distillation/run_pipeline.py
```

### 5. Interactive Chat

```bash
python 04_distillation/chat.py
```

### 6. Evaluate Results

```bash
python -m 05_evaluation.clarity_judge --input results.json --model qwen2.5:7b-instruct
```

### Running Tests

```bash
pytest tests/ -v
```

## Technical Details

### Distillation Approach

The project uses **logits-based distillation** with a hybrid loss function:

```
L = alpha * L_soft + (1 - alpha) * L_hard
```

Where:
- `L_soft`: KL divergence between teacher and student logits (with temperature scaling)
- `L_hard`: Cross-entropy on actual labels
- `alpha = 0.7` (default)

### Models

| Role    | Model                              | Parameters |
|---------|------------------------------------|-----------:|
| Teacher | meta-llama/Llama-2-7b-chat-hf     |       ~7B  |
| Student | TinyLlama/TinyLlama-1.1B-Chat-v1.0|     ~1.1B  |

### RAG Stack

- **Embeddings**: BAAI/bge-m3 (multilingual, optimized for retrieval)
- **Vector DB**: ChromaDB with persistent storage
- **Reranker**: BAAI/bge-reranker-v2-m3 (two-stage retrieval)
- **Chunking**: 800 tokens, 150 overlap (RecursiveCharacterTextSplitter)

### Evaluation Dimensions

The LLM-as-Judge evaluates on 4 weighted dimensions:
- **Accuracy** (40%): Factual correctness
- **Relevance** (30%): Answer pertinence to the question
- **Completeness** (20%): Coverage of key points
- **Clarity** (10%): Readability and structure

## License

MIT License - see [LICENSE](LICENSE) for details.
