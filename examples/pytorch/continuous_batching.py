import time

import datasets
import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# --- Common Setup ---
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3b-Instruct", attn_implementation="sdpa_paged", torch_dtype=torch.bfloat16, device_map="auto"
).eval()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3b-Instruct", padding_side="left")

device = "cuda"
model.use_cache = False
# Set pad token if missing (common for Llama models)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Configure generation parameters
generation_config = GenerationConfig(
    max_new_tokens=10,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
    use_cache=False,
    num_blocks=2048,
    block_size=128,
    max_batch_tokens=512,  # Maximum number of tokens to process in a single batch
)

# Prepare data (using a smaller subset for demonstration)
train_dataset = datasets.load_dataset("openai/gsm8k", "socratic", split="test")
# train_dataset = train_dataset.select(range(5))  # Use only 5 examples for the simple version

# --- Example 1: Simple Version using generate_batch ---
print("--- Running CB Generation Example ---")
model.config.attn_implementation = "paged_attention"
def tokenize_function(examples):
    # Truncate to avoid overly long prompts exceeding max context length
    return tokenizer(examples["question"])


tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
simple_batch_inputs = [item["input_ids"] for item in tokenized_datasets]
model.__call__ = torch.compile(model.__call__, mode="max-autotune-no-cudagraphs", fullgraph=True)


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
