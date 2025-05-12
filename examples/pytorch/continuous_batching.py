import time

import datasets
import torch
from tokenizers.decoders import DecodeStream

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


_TEST_PROMPTS = [
    "Describe a fruit that is of orange color and round. It is a sweet fruit and a great source of Vitamine C. The fruit I'm thinking of is an",
    "A man is a walking his dog down the street, and a the turn he sees",
    "A plane is flying high in the sky, out of the window are clouds and mountains. Where could the plane be located?",
    "Please fill in the form to",
    "For safety reasons, the train is stopped in the middle of the",
]

# --- Common Setup ---
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3b-Instruct", attn_implementation="sdpa_paged", torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3b-Instruct", torch_dtype=torch.float16, padding_side="left"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set pad token if missing (common for Llama models)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Configure generation parameters
generation_config = GenerationConfig(
    max_new_tokens=16,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    # Add other parameters like temperature, top_k etc. if needed
    # Example:
    # temperature=0.7,
    # top_k=50,
    # Parameters relevant for Continuous Batching (can be tuned)
    num_blocks=8,
    block_size=1024,
    max_batch_tokens=512,  # Maximum number of tokens to process in a single batch
)

# Prepare data (using a smaller subset for demonstration)
train_dataset = datasets.load_dataset("imdb", split="test")
train_dataset = train_dataset.select(range(5))  # Use only 5 examples for the simple version


def tokenize_function(examples):
    # Truncate to avoid overly long prompts exceeding max context length
    return tokenizer(examples["text"], truncation=True, max_length=512)


# tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
# simple_batch_inputs = [item["input_ids"] for item in tokenized_test_prompts]
tokenized_test_prompts = tokenizer(_TEST_PROMPTS, truncation=True, max_length=512)
simple_batch_inputs = list(tokenized_test_prompts["input_ids"])

model.config.attn_implementation = "sdpa"
start_time_simple = time.time()
outputs = model.generate(
    input_ids=tokenizer(_TEST_PROMPTS, truncation=True, max_length=512, padding=True, return_tensors="pt").input_ids.to(model.device),
    generation_config=GenerationConfig(
    max_new_tokens=25,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id),
    # You can pass request-specific overrides here, e.g., max_new_tokens=100
)
end_time_simple = time.time()
print(f"\nSimple batch generation took: {end_time_simple - start_time_simple:.2f} seconds")

print("\nResults from simple generate_batch:")
for i, request in enumerate(outputs):
    output_text = tokenizer.decode(request, skip_special_tokens=False)
    print("-" * 20)
    print(f"Result for Request {request}:")
    print(f"  Output: {output_text}")
print("-" * 20)
print("--- Finished Simple Batch Generation Example ---\n\n")

# --- Example 1: Simple Version using generate_batch ---
print("--- Running Simple Batch Generation Example ---")

start_time_simple = time.time()
# Call the simple batch generation function
# This handles manager initialization, request adding, result retrieval, and shutdown internally.
batch_outputs = model.generate_batch(
    inputs=simple_batch_inputs,
    generation_config=generation_config,
    # You can pass request-specific overrides here, e.g., max_new_tokens=100
)
end_time_simple = time.time()

print(f"generation config: {generation_config}")

print(f"\nSimple batch generation took: {end_time_simple - start_time_simple:.2f} seconds")

# Decode and print results
print("\nResults from simple generate_batch:")
for request in batch_outputs:
    input_text = tokenizer.decode(batch_outputs[request].prompt_ids, skip_special_tokens=False)
    output_text = tokenizer.decode(batch_outputs[request].static_outputs, skip_special_tokens=False)
    print("-" * 20)
    print(f"Result for Request {request}:")
    print(f"  Input:  {input_text}")
    print(f"  Output: {output_text}")
print("-" * 20)
print("--- Finished Simple Batch Generation Example ---\n\n")

