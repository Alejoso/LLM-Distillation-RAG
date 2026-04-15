from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    "seed": 42,

    "teacher_name": "meta-llama/Llama-2-7b-chat-hf",
    "student_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",

    "epochs": 1,
    "batch_size": 1,
    "lr": 2e-5,
    "temperature": 4.0,
    "alpha": 0.7,
    "max_len": 64,
    "max_new_tokens": 32,

    "save_steps": 1000,

    # Dataset paths
    "raw_data_path": BASE_DIR / "data" / "raw_train.json",
    "processed_path": BASE_DIR / "data" / "processed.json",
    "distilled_data_path": BASE_DIR / "data" / "train.jsonl",

    # Output paths
    "output_dir": BASE_DIR / "outputs",
    "checkpoint_dir": BASE_DIR / "outputs" / "checkpoints",
    "checkpoint_path": BASE_DIR / "outputs" / "checkpoints" / "last.pt",
    "log_file": BASE_DIR / "outputs" / "logs.jsonl",
    "final_model_dir": BASE_DIR / "outputs" / "final_model",

    # Teacher behavior
    "use_rag_for_teacher": True,

    # RAG config
    "rag_chroma_path": BASE_DIR / "chroma",
    "rag_top_k_initial": 10,
    "rag_top_k_final": 6,
}


def ensure_directories():
    (BASE_DIR / "data").mkdir(exist_ok=True)
    CONFIG["output_dir"].mkdir(exist_ok=True)
    CONFIG["checkpoint_dir"].mkdir(exist_ok=True)