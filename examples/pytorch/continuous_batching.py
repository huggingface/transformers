import time
import argparse
import datasets
import torch
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


def batch_generate(
    model: AutoModelForCausalLM,
    simple_batch_inputs: list,
    generation_config: GenerationConfig,
    tokenizer: AutoTokenizer,
    displayed_samples: int = 0, # -1: no display, 0: display stats, >0: display inputs and some outputs
    output_file: str = None,
) -> tuple[float, float]:

    # Actual batch generation
    if displayed_samples >= 0:
        print("--- Running CB Generation Example ---")
    start_time_simple = time.time()
    batch_outputs = model.generate_batch(
        inputs=simple_batch_inputs,
        generation_config=generation_config,
    )
    end_time_simple = time.time()
    if displayed_samples >= 0:
        print("Done with batch generation.")

    # Decode outputs
    token_count = 0
    data = []
    for i, request in enumerate(batch_outputs):
        input_text = tokenizer.decode(batch_outputs[request].prompt_ids, skip_special_tokens=False)
        data.append({"input": input_text})
        try:
            output_text = tokenizer.decode(batch_outputs[request].generated_tokens, skip_special_tokens=False)
            token_count += len(batch_outputs[request].generated_tokens[1:])
            data[-1]["output"] = output_text
        except Exception as e:
            print(f"Decoding failed for request {request}: {e}")
            data[-1]["output"] = "__ERROR__"
        if i < displayed_samples:
            if len(output_text) > 0:
                print("-" * 20)
                print(f"{request} Input:  {input_text}")
                print(f"{request} Output: {output_text}")
            else:
                print(f"{request} Input:  {input_text}")
                print("[WARN]")
                print(f"{request} Output was empty!")

    # If an output file is provided, save the reordered data to it
    data.sort(key=lambda x: x["input"])
    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)

    # Compute stats and maybe print them
    gen_time = end_time_simple - start_time_simple
    tok_per_sec = token_count / gen_time
    if displayed_samples >= 0:
        print("-" * 20)
        print("--- Finished CB Generation Example ---\n")
        print(f"CB generation took: {gen_time:.2f} seconds for {token_count} tokens. {tok_per_sec:.2f}tok/s")
    return gen_time, tok_per_sec


if __name__ == "__main__":

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--max-batch-tokens", type=int, default=None)

    parser.add_argument("--attn", type=str, default="paged_attention|kernels-community/flash-attn", help="Attention implementation")
    parser.add_argument("--matmul-precision", "-mp", type=str, default="high") # set to "none" to disable
    parser.add_argument("--use-cuda-graph", action="store_true", default=False)

    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--displayed", type=int, default=1, help="Number of samples to display")
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()

    # Set matmul precision
    if args.matmul_precision != "none":
        torch.set_float32_matmul_precision(args.matmul_precision)

    # Prepare model
    model_id = "meta-llama/Llama-3.2-3b-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation=args.attn,
        dtype=torch.bfloat16,
        torch_dtype=torch.bfloat16,
    )
    model = model.cuda().eval()
    # model.forward = torch.compile(model.forward, mode="max-autotune-no-cudagraphs")

    # Prepare tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    dataset = datasets.load_dataset("openai/gsm8k", "socratic", split="test")
    dataset = dataset.select(range(args.samples))  # Use only 5 examples for the simple version
    tokenized_datasets = dataset.map(lambda x: tokenizer(x["question"]), batched=True)
    simple_batch_inputs = [item["input_ids"] for item in tokenized_datasets]

    # Prepare generation config
    generation_config = GenerationConfig(
        max_new_tokens=512,
        use_cuda_graph=args.use_cuda_graph,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        num_blocks=args.num_blocks,
        max_batch_tokens=args.max_batch_tokens,
    )

    # Run warmup batch generation
    batch_generate(
        model,
        simple_batch_inputs[:min(5, args.samples)],
        generation_config,
        tokenizer,
        displayed_samples=-1,
    )

    # Run batch generation
    gen_time, tok_per_sec = batch_generate(
        model,
        simple_batch_inputs,
        generation_config,
        tokenizer,
        displayed_samples=args.displayed,
        output_file=args.output_file,
    )


# python examples/pytorch/continuous_batching.py --attn sdpa_paged --matmul-precision none --samples 50 --displayed 0
# Using calculated self.num_blocks = 4096, self.block_size = 32, self.max_batch_tokens = 2048
# CB generation took: 18.80 seconds for 13775 tokens. 732.74tok/s


# python examples/pytorch/continuous_batching.py --attn sdpa_paged --matmul-precision none --samples 100 --displayed 1
# Setting up static tensors with T = 4096, max_token_budget = 524288, 139538202624 bytes available
# CB generation took: 29.53 seconds for 26384 tokens. 893.41tok/s
