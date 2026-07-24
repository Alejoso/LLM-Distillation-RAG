import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_teacher_and_student(config):
    device = get_device()
    print(f"Device: {device}")

    teacher_tokenizer = load_tokenizer(config["teacher_name"])
    student_tokenizer = load_tokenizer(config["student_name"])

    teacher = AutoModelForCausalLM.from_pretrained(
        config["teacher_name"]
    ).to(device)

    student = AutoModelForCausalLM.from_pretrained(
        config["student_name"]
    ).to(device)

    teacher.eval()

    return teacher_tokenizer, student_tokenizer, teacher, student, device


def load_final_model(config):
    device = get_device()
    model_path = str(config["final_model_dir"])

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    return tokenizer, model, device