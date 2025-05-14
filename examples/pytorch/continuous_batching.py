import time

import datasets
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


_TEST_PROMPTS = [
    "A man is a walking his dog down the street, and a the turn he sees",
    "Describe a fruit that is of orange color and round. It is a sweet fruit and a great source of Vitamine C. The fruit I'm thinking of is an",
    "A plane is flying high in the sky, out of the window are clouds and mountains. Where could the plane be located?",
    "Please fill in the form to",
    "For safety reasons, the train is stopped in the middle of the",
]

# --- Common Setup ---
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3b-Instruct", attn_implementation="sdpa", torch_dtype=torch.bfloat16, device_map="auto"
).eval()
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3b-Instruct", padding_side="left"
)

device = "mps"
model.use_cache = False
# Set pad token if missing (common for Llama models)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Configure generation parameters
generation_config = GenerationConfig(
    max_new_tokens=50,
    top_k=0,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    use_cache=False,
    num_blocks=512,
    block_size=128,
    max_batch_tokens=512,  # Maximum number of tokens to process in a single batch
)

# Prepare data (using a smaller subset for demonstration)
train_dataset = datasets.load_dataset("openai/gsm8k", "socratic", split="test")
# train_dataset = train_dataset.select(range(100))  # Use only 5 examples for the simple version

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

# --- Example 1: Simple Version using generate_batch ---
print("--- Running CB Generation Example ---")
model.config.attn_implementation = "paged_attention"
def tokenize_function(examples):
    # Truncate to avoid overly long prompts exceeding max context length
    return tokenizer(examples["question"])


tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
simple_batch_inputs = [item["input_ids"] for item in tokenized_datasets]


# tokenized_test_prompts = tokenizer(_TEST_PROMPTS, truncation=True, max_length=512)
# simple_batch_inputs = list(tokenized_test_prompts["input_ids"])

start_time_simple = time.time()
# Call the simple batch generation function
# This handles manager initialization, request adding, result retrieval, and shutdown internally.
batch_outputs = model.generate_batch(
    inputs=simple_batch_inputs,
    generation_config=generation_config,
    # You can pass request-specific overrides here, e.g., max_new_tokens=100
)
end_time_simple = time.time()
model.__call__ = torch.compile(model.__call__, mode="reduce-overhead")

print(f"CB generation took: a{end_time_simple - start_time_simple:.2f} seconds")

# Decode and print results
print("\nResults from simple generate_batch:")
for request in batch_outputs:
    input_text = tokenizer.decode(batch_outputs[request].full_prompt_ids, skip_special_tokens=False)
    try:
        # Decode the static outputs
        output_text = tokenizer.decode(batch_outputs[request].static_outputs, skip_special_tokens=False)
    except Exception as e:
        # Handle the case where decoding fails
        print(f"Decoding failed for request {request}: {e}")
        output_text = tokenizer.decode(batch_outputs[request].static_outputs[1:], skip_special_tokens=False)
    if len(output_text) > 0:
        print("-" * 20)
        print(f"{request} Input:  {input_text}")
        print(f"{request} Output: {output_text}")
    else:
        print("", end="\r\r\r\r")
print("-" * 20)
print("--- Finished Simple Batch Generation Example ---\n\n")
