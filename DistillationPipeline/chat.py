from config import CONFIG
from src.inference import ask_model
from src.models import load_final_model


def main():
    tokenizer, model, device = load_final_model(CONFIG)
    print("Model loaded successfully. Type 'exit' to quit.\n")

    while True:
        instruction = input("Instruction: ").strip()
        if instruction.lower() in {"exit", "quit"}:
            break

        context = input("Context (optional): ").strip()
        response = ask_model(
            model=model,
            tokenizer=tokenizer,
            device=device,
            instruction=instruction,
            context=context,
            max_new_tokens=CONFIG["max_new_tokens"]
        )

        print("\nResponse:")
        print(response)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()