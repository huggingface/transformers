import time

import datasets
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


torch.set_float32_matmul_precision("high")

model_id = "meta-llama/Llama-3.2-3b-Instruct"
model = (
    AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="paged_attention|kernels-community/flash-attn",
        dtype=torch.bfloat16,
    )
    .eval()
    .cuda()
)
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

generation_config = GenerationConfig(
    max_new_tokens=512,
    # use_cuda_graph=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=False,
)

train_dataset = datasets.load_dataset("openai/gsm8k", "socratic", split="test")
train_dataset = train_dataset.select(range(500))  # Use only 5 examples for the simple version
print("--- Running CB Generation Example ---")


def tokenize_function(examples):
    return tokenizer(examples["question"])


tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
simple_batch_inputs = [item["input_ids"] for item in tokenized_datasets]

start_time_simple = time.time()
model.forward = torch.compile(model.forward, mode="max-autotune-no-cudagraphs")
batch_outputs = model.generate_batch(
    inputs=simple_batch_inputs,
    generation_config=generation_config,
)
end_time_simple = time.time()
token_count = 0
for request in batch_outputs:
    input_text = tokenizer.decode(batch_outputs[request].prompt_ids, skip_special_tokens=False)
    try:
        output_text = tokenizer.decode(batch_outputs[request].generated_tokens, skip_special_tokens=False)
        token_count += len(batch_outputs[request].generated_tokens[1:])
    except Exception as e:
        print(f"Decoding failed for request {request}: {e}")
        token_count += len(batch_outputs[request].generated_tokens[1:])
        output_text = tokenizer.decode(batch_outputs[request].generated_tokens[1:], skip_special_tokens=False)
    if len(output_text) > 0:
        print("-" * 20)
        print(f"{request} Input:  {input_text}")
        print(f"{request} Output: {output_text}")
    else:
        print("", end="\r\r\r\r")
print("-" * 20)
print("--- Finished CB Generation Example ---\n\n")


print(
    f"CB generation took: {end_time_simple - start_time_simple:.2f} seconds for {token_count} tokens. {token_count / (end_time_simple - start_time_simple)}tok/s"
)


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
