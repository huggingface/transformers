import datasets
import torch
import time
import queue

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
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3b-Instruct", attn_implementation="sdpa", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3b-Instruct", torch_dtype=torch.float16, padding_side="left")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set pad token if missing (common for Llama models)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Configure generation parameters
generation_config = GenerationConfig(
    max_new_tokens=50,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    # Add other parameters like temperature, top_k etc. if needed
    # Example:
    # temperature=0.7,
    # top_k=50,
    # Parameters relevant for Continuous Batching (can be tuned)
    batch_size=8, # Internal micro-batch size for the processor
    num_blocks=8,
    block_size=1024
)

# Prepare data (using a smaller subset for demonstration)
train_dataset = datasets.load_dataset('imdb', split='test')
train_dataset = train_dataset.select(range(5)) # Use only 5 examples for the simple version

def tokenize_function(examples):
    # Truncate to avoid overly long prompts exceeding max context length
    return tokenizer(examples["text"], truncation=True, max_length=512)

# tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
# simple_batch_inputs = [item["input_ids"] for item in tokenized_test_prompts]
tokenized_test_prompts = tokenizer(_TEST_PROMPTS, truncation=True, max_length=512)
simple_batch_inputs = [item for item in tokenized_test_prompts["input_ids"]]


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
for i, output_ids in enumerate(batch_outputs):
    input_text = tokenizer.decode(simple_batch_inputs[i], skip_special_tokens=False)
    output_text = tokenizer.decode(output_ids, skip_special_tokens=False)
    print("-" * 20)
    print(f"Result for Request {i}:")
    # print(f"  Input:  {input_text}")
    print(f"  Output: {output_text}")
print("-" * 20)
print("--- Finished Simple Batch Generation Example ---\n\n")

outputs = []

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3b-Instruct", attn_implementation="sdpa", torch_dtype=torch.float16, device_map="auto")

print("--- Running Simple Generation for comparison ---")
# tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
# simple_inputs = [torch.tensor(item["input_ids"], device=device) for item in tokenized_test_prompts]
tokenized_test_prompts = tokenizer(_TEST_PROMPTS, truncation=True, max_length=512)
simple_inputs = [torch.tensor(item, device=device) for item in tokenized_test_prompts["input_ids"]]


padded_inputs = tokenizer.pad({"input_ids": [item for item in tokenized_test_prompts["input_ids"]]}, return_tensors="pt").to(device)


start_time_simple = time.time()
outputs = model.generate(
    **padded_inputs,
    generation_config=generation_config,
    do_sample=False,
)
end_time_simple = time.time()

print(f"generation config: {generation_config}")

print(f"\nSimple generation took: {end_time_simple - start_time_simple:.2f} seconds")

print("\nResults from simple generate:")
for i, output_ids in enumerate(outputs):
    input_text = tokenizer.decode(simple_inputs[i], skip_special_tokens=False)
    # The output_ids from batch generation include the input tokens, skip them for decoding
    # We need to know the length of the input to slice the output correctly
    input_length = len(simple_inputs[i])
    # Slice the output ids to get only the generated part
    generated_ids = output_ids[input_length:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    print("-" * 20)
    print(f"Result for Request {i}:")
    # print(f"  Input:  {input_text}")
    print(f"  Output: {output_text}")
print("-" * 20)

print("--- Finished Simple Generation Example ---\n\n")

# --- Example 2: Streaming Version using ContinuousBatchingManager ---
print("--- Running Streaming Continuous Batching Example ---")

manager = model.init_continuous_batching(generation_config=generation_config, streaming=True)

manager.start()

req_id = manager.add_request(simple_batch_inputs[0])

request_streams = {}

first_token = True
for output in manager:
    req_id = output["request_id"]
    if req_id is None:
        continue
    if first_token:
        print(f"Request {req_id} started")
        first_token = False
    if req_id not in request_streams:
        request_streams[req_id] = DecodeStream(skip_special_tokens=False)
    next_token = request_streams[req_id].step(tokenizer._tokenizer, output["next_token"])
    print(f"{next_token}", end="")
    if output["status"] in ["finished", "failed"]:
        print(f"\nRequest {req_id} {output['status']}")
        del request_streams[req_id]
        break

manager.stop()
manager.join(timeout=10)

import sys
sys.exit(0)

# --- Example 3: Involved Performant Version using ContinuousBatchingManager ---
print("--- Running Involved Continuous Batching Example ---")

# Prepare data for the involved example (using a larger dataset)
involved_dataset = datasets.load_dataset('imdb', split='test')
involved_dataset = involved_dataset.select(range(20)) # Use 20 examples
tokenized_involved_datasets = involved_dataset.map(tokenize_function, batched=True)
# Extract input_ids
requests_data = [{"input_ids": item["input_ids"]} for item in tokenized_involved_datasets]

# 1. Initialize the manager
manager = model.init_continuous_batching(generation_config=generation_config)

# Optional: Provide initial shapes to help cache calculation (if using many similar length prompts)
# manager.add_initial_prompts([req['input_ids'] for req in requests_data[:5]])

# 2. Start the background generation thread
manager.start()

submitted_requests = {}
results = {}
start_time_involved = time.time() # Start timing before submission

for i, req_data in enumerate(requests_data):
    try:
        req_id = manager.add_request(req_data["input_ids"], request_id=f"req_{i}")
        submitted_requests[req_id] = {"input": tokenizer.decode(req_data["input_ids"])}
        print(f"Submitted request {req_id}")
    except Exception as e:
        print(f"Failed to submit request {i}: {e}")


# 3. Retrieve results
finished_count = 0
while finished_count < len(submitted_requests):
    try:
        result = manager.get_result(timeout=1.0)
        if result:
            req_id = result["request_id"]
            finished_count += 1
            results[req_id] = result
            output_text = tokenizer.decode(result["output_ids"], skip_special_tokens=True)
            print("-" * 20)
            print(f"Result for {req_id} (Status: {result['status']}):")
            # print(f"  Input:  {submitted_requests[req_id]['input'][:100]}...") # Optional: print input
            print(f"  Output: {output_text}")
            print("-" * 20)
    except queue.Empty:
        if not manager.is_running():
            print("Manager thread stopped, but not all results received. Exiting retrieval.")
            break
    except Exception as e:
        print(f"Error retrieving results: {e}")

end_time_involved = time.time() # End timing after retrieval loop
print(f"\nInvolved continuous batching took: {end_time_involved - start_time_involved:.2f} seconds (includes submission delays)")

print(f"Total submitted: {len(submitted_requests)}")
print(f"Total results received: {len(results)}")

print("Stopping the manager...")
manager.stop()
manager.join(timeout=10) # Wait for the thread to exit

print("Manager stopped.")

# You can now process the `results` dictionary which contains
# {"request_id": ..., "output_ids": ..., "status": ...} for each finished request.

print("--- Finished Advanced Continuous Batching Example ---")


