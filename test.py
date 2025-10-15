import argparse
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Quick NanoGPT chat inference test")
    parser.add_argument(
        "--model",
        default="chat-d20",
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

    rendered = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    print(rendered)

    model_inputs = tokenizer([rendered], return_tensors="pt").to(device)
    if "attention_mask" not in model_inputs:
        model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"]).to(device)

    with torch.no_grad():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = generated[0, model_inputs.input_ids.shape[1] :]
    print(tokenizer.decode(output_ids, skip_special_tokens=True))
    print("--------------------------------")
    print(tokenizer.decode(output_ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
