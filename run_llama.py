import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
import lovely_tensors as lt; lt.monkey_patch()

class Colors:
    """ANSI color codes"""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"


if __name__ == "__main__":
    if torch.distributed.is_available() and "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = 0
    parser = argparse.ArgumentParser(description="Run inference with or without tensor parallelism.")
    parser.add_argument(
        "--tp_size",
        help="Size of the tensor parallelism.",
        type=int,
    )
    parser.add_argument(
        "--pp_size",
        help="Size of the pipeline parallelism.",
        type=int,
    )
    args = parser.parse_args()

    #TOOD(3outeille): Refacto so that we can handle both without specifying tp and pp size
    if args.tp_size:
        device_map_args = {"tp_plan": "auto", "tp_size": args.tp_size, "pp_size": 1}
        if rank == 0:
            print(f"Running with tensor parallelism enabled (tp_size={args.tp_size}).")
        prompt_color = Colors.BLUE
        generated_color = Colors.GREEN
    elif args.pp_size:
        device_map_args = {"pp_plan": "auto", "pp_size": args.pp_size, "tp_size": 1}
        if rank == 0:   
            print(f"Running with pipeline parallelism enabled (pp_size={args.pp_size}).")
        prompt_color = Colors.BLUE
        generated_color = Colors.YELLOW
    else:
        device_map_args = {}
        if rank == 0:
            print("Running on a single device (no parallelism).")
        prompt_color = Colors.BLUE
        generated_color = Colors.RED

    torch.manual_seed(42)

    model_id = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        **device_map_args,
    )

    prompt = "The capital of France is"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 6. Generate output tokens
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        temperature=0.6,
        top_p=0.9,
    )

    response_ids = outputs[0][inputs.input_ids.shape[-1]:]
    generated_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    if rank == 0:
        print(f"\nPrompt: {prompt_color}'{prompt}'{Colors.ENDC}")
        print(f"Generated: {generated_color}'{generated_text}'{Colors.ENDC}")