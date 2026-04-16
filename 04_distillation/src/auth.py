import os
from huggingface_hub import login


def hf_login_if_needed():
    token = os.getenv("HF_TOKEN")

    if token:
        login(token=token, add_to_git_credential=False)
        print("Hugging Face login successful using HF_TOKEN.")
    else:
        print("HF_TOKEN not found. Using cached/local Hugging Face credentials if available.")