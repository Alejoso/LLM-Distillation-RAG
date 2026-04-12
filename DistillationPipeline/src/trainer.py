import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import DistillationDataset


def train_student(config, student, student_tokenizer, device):
    dataset = DistillationDataset(
        path=config["distilled_data_path"],
        tokenizer=student_tokenizer,
        max_len=config["max_len"]
    )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=config["lr"])

    checkpoint_path = Path(config["checkpoint_path"])
    log_file = Path(config["log_file"])
    final_model_dir = Path(config["final_model_dir"])

    start_epoch = 0
    global_step = 0

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        student.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["step"]
        print("Checkpoint loaded.")

    student.train()

    alpha = config["alpha"]
    temperature = config["temperature"]

    for epoch in range(start_epoch, config["epochs"]):
        epoch_loss = 0.0

        for batch in loader:
            global_step += 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            teacher_logits = batch["teacher_logits"].to(device)
            response_mask = batch["response_mask"].to(device)

            outputs = student(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            student_logits = outputs.logits

            shift_student = student_logits[:, :-1, :]
            shift_teacher = teacher_logits[:, 1:, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = response_mask[:, 1:]

            teacher_probs = F.softmax(shift_teacher / temperature, dim=-1)
            student_log_probs = F.log_softmax(shift_student / temperature, dim=-1)

            soft_token_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="none"
            ).sum(dim=-1)

            soft_loss = (soft_token_loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)
            soft_loss = soft_loss * (temperature ** 2)

            hard_token_loss = F.cross_entropy(
                shift_student.reshape(-1, shift_student.size(-1)),
                shift_labels.reshape(-1),
                reduction="none"
            ).view_as(shift_labels)

            hard_loss = (hard_token_loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)

            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN/Inf detected, skipping batch.")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if global_step % config["save_steps"] == 0:
                torch.save({
                    "model": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": global_step
                }, checkpoint_path)

            log = {
                "epoch": epoch,
                "step": global_step,
                "loss": loss.item(),
                "soft_loss": soft_loss.item(),
                "hard_loss": hard_loss.item(),
                "time": time.time()
            }

            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")

        avg_epoch_loss = epoch_loss / max(len(loader), 1)
        print(
            f"Epoch {epoch + 1}/{config['epochs']} | "
            f"avg_loss: {avg_epoch_loss:.4f} | "
            f"soft: {soft_loss.item():.4f} | "
            f"hard: {hard_loss.item():.4f}"
        )

    final_model_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(str(final_model_dir))
    student_tokenizer.save_pretrained(str(final_model_dir))

    print(f"Final model saved to: {final_model_dir}")