import torch

from config import CONFIG, ensure_directories
from src.auth import hf_login_if_needed
from src.data_processing import process_dataset
from src.distill_data import generate_distillation_jsonl
from src.models import load_teacher_and_student
from src.trainer import train_student
from src.utils import set_seed


def main():
    ensure_directories()
    set_seed(CONFIG["seed"])
    hf_login_if_needed()

    process_dataset(CONFIG)

    teacher_tokenizer, student_tokenizer, teacher, student, device = load_teacher_and_student(CONFIG)

    generate_distillation_jsonl(CONFIG, teacher, teacher_tokenizer, device)

    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_student(CONFIG, student, student_tokenizer, device)


if __name__ == "__main__":
    main()