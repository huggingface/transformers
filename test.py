import argparse
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Quick NanoGPT chat inference test")
    parser.add_argument(
        "--model",
        default="nanochat-students/d20-chat-transformers",
        help="Model path or repo id to load (default: local chat-d20 directory)",
    )
    parser.add_argument(
        "--prompt",
        default="What is the capital of France?",
        help="User message for the conversation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = pathlib.Path(args.model)
    if model_path.is_dir():
        model_id = str(model_path.resolve())
    else:
        model_id = args.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=False, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    conversation = [
        {"role": "user", "content": args.prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=args.max_new_tokens,
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("--------------------------------")
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()
