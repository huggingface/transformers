import time
import argparse
import datasets
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


def batch_generate(
    model: AutoModelForCausalLM,
    simple_batch_inputs: list,
    generation_config: GenerationConfig,
    tokenizer: AutoTokenizer,
    displayed_samples: int = 0, # -1: no display, 0: display stats, >0: display inputs and some outputs
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
    for i, request in enumerate(batch_outputs):
        input_text = tokenizer.decode(batch_outputs[request].prompt_ids, skip_special_tokens=False)
        try:
            output_text = tokenizer.decode(batch_outputs[request].generated_tokens, skip_special_tokens=False)
            token_count += len(batch_outputs[request].generated_tokens[1:])
        except Exception as e:
            print(f"Decoding failed for request {request}: {e}")
            token_count += len(batch_outputs[request].generated_tokens[1:])
            output_text = tokenizer.decode(batch_outputs[request].generated_tokens[1:], skip_special_tokens=False)
        if i < displayed_samples:
            if len(output_text) > 0:
                print("-" * 20)
                print(f"{request} Input:  {input_text}")
                print(f"{request} Output: {output_text}")
            else:
                print(f"{request} Input:  {input_text}")
                print("[WARN]")
                print(f"{request} Output was empty!")

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
    parser.add_argument("--attn-implementation", type=str, default="paged_attention|kernels-community/flash-attn")
    parser.add_argument("--matmul-precision", type=str, default="high") # set to "none" to disable
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--use-cuda-graph", action="store_true")
    args = parser.parse_args()

    # Set matmul precision
    if args.matmul_precision != "none":
        torch.set_float32_matmul_precision(args.matmul_precision)

    # Prepare model
    model_id = "meta-llama/Llama-3.2-3b-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation=args.attn_implementation,
        dtype=torch.bfloat16,
        torch_dtype=torch.bfloat16,
    )
    model = model.cuda().eval()
    model.forward = torch.compile(model.forward, mode="max-autotune-no-cudagraphs")

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
        displayed_samples=5,
    )


# TODO: remove this or incorporate it into the script above

# train_dataset = train_dataset.select(range(5))  # Use only 5 examples for the simple version

# tokenized_test_prompts = tokenizer(_TEST_PROMPTS, padding=True, padding_side="left", truncation=True, max_length=512)
# simple_batch_inputs = list(tokenized_test_prompts["input_ids"])

# def tokenize_function(examples):
#     # Truncate to avoid overly long prompts exceeding max context length
#     return tokenizer(examples["question"], padding=True, truncation=True, max_length=512)


# tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
# simple_batch_inputs = [item["input_ids"] for item in tokenized_datasets]


# model.config.attn_implementation = "sdpa"
# start_time_simple = time.time()
# batch_size = 64
# full_outputs = []
# from tqdm import tqdm

# for i in tqdm(range(0, len(simple_batch_inputs)-batch_size, batch_size)):
#     outputs = model.generate(
#         torch.tensor(simple_batch_inputs[i:i+batch_size], device=model.device),
#         generation_config=GenerationConfig(
#             max_new_tokens=16, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id
#         ),
#     )
#     full_outputs.extend(outputs.tolist())

# end_time_simple = time.time()
# print(f"\nSimple batch generation took: {end_time_simple - start_time_simple:.2f} seconds")

# print("\nResults from simple generate_batch:")
# for i, request in enumerate(full_outputs):
#     output_text = tokenizer.decode(request, skip_special_tokens=False)
#     print("-" * 20)
#     print(f"  Output: {output_text}")
# print("-" * 20)
# print("--- Finished Simple Batch Generation Example ---\n\n")
