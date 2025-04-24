import datasets
import torch
import time
import threading
import queue

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


# --- Common Setup ---
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", attn_implementation="sdpa", torch_dtype=torch.float16, device_map="auto")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", attn_implementation="eager", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16)

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

tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
# Extract input_ids for the simple batch
simple_batch_inputs = [item["input_ids"] for item in tokenized_datasets]


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

print(f"\nSimple batch generation took: {end_time_simple - start_time_simple:.2f} seconds")

# Decode and print results
print("\nResults from simple generate_batch:")
for i, output_ids in enumerate(batch_outputs):
    input_text = tokenizer.decode(simple_batch_inputs[i], skip_special_tokens=True)
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print("-" * 20)
    print(f"Result for Request {i}:")
    # print(f"  Input:  {input_text[:100]}...") # Optional: print input
    print(f"  Output: {output_text}")
print("-" * 20)
print("--- Finished Simple Batch Generation Example ---\n\n")

# --- Example 2: Involved Performant Version using ContinuousBatchingManager ---
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

# 3. Thread to add requests
def add_requests_thread():
    print("Starting request submission thread...")
    for i, req_data in enumerate(requests_data):
        try:
            # Add other generation params per request if needed, e.g., max_new_tokens
            req_id = manager.add_request(req_data["input_ids"], request_id=f"req_{i}")
            submitted_requests[req_id] = {"input": tokenizer.decode(req_data["input_ids"])}
            print(f"Submitted request {req_id}")
            time.sleep(0.1) # Simulate requests arriving over time
        except Exception as e:
            print(f"Failed to submit request {i}: {e}")
            # Handle submission failure if needed
    print(f"Finished submitting {len(submitted_requests)} requests.")

# 4. Main thread (or another thread) to retrieve results
def retrieve_results():
    print("Starting results retrieval...")
    finished_count = 0
    while finished_count < len(submitted_requests):
        try:
            result = manager.get_result(timeout=1.0) # Wait for 1 second
            if result:
                req_id = result["request_id"]
                if req_id in submitted_requests:
                    results[req_id] = result
                    finished_count += 1
                    output_text = tokenizer.decode(result["output_ids"], skip_special_tokens=True)
                    print("-" * 20)
                    print(f"Result for {req_id} (Status: {result['status']}):")
                    # print(f"  Input:  {submitted_requests[req_id]['input'][:100]}...") # Optional: print input
                    print(f"  Output: {output_text}")
                    print("-" * 20)
                else:
                    print(f"Received result for unknown request ID: {req_id}")
            # Add a small sleep if no result to prevent busy-waiting if timeout=0 used
            # time.sleep(0.01)
        except queue.Empty:
            # Timeout occurred, check if the manager is still running
            # Accessing _generation_thread directly is internal, consider adding a is_running() method to manager
            if not manager.is_running():
                 print("Manager thread stopped, but not all results received. Exiting retrieval.")
                 break
            continue # Continue waiting
        except Exception as e:
             print(f"Error retrieving results: {e}")
             # Decide if retrieval should stop on error
             break
    print(f"Finished retrieving {finished_count} results.")
    end_time_involved = time.time() # End timing after retrieval loop
    print(f"\nInvolved continuous batching took: {end_time_involved - start_time_involved:.2f} seconds (includes submission delays)")


# Start the submission thread
submit_thread = threading.Thread(target=add_requests_thread)
submit_thread.start()

# Retrieve results in the main thread
retrieve_results()

# Wait for submission thread to finish (optional, retrieve_results loop handles waiting for results)
submit_thread.join()

# 5. Stop the manager
print("Stopping the manager...")
manager.stop()
manager.join(timeout=10) # Wait for the thread to exit

print("Manager stopped.")

# Final check for any missed results (if stop was called early)
print("Checking for any remaining results in queue...")
while True:
    try:
        result = manager.get_result(timeout=1.0)
        if result:
             req_id = result["request_id"]
             if req_id not in results: # Avoid printing duplicates
                 results[req_id] = result
                 print(f"Found remaining result for {req_id}: {tokenizer.decode(result['output_ids'])}")
        else:
            break # Queue is empty
    except queue.Empty:
        break

print(f"Total submitted: {len(submitted_requests)}")
print(f"Total results received: {len(results)}")

# You can now process the `results` dictionary which contains
# {"request_id": ..., "output_ids": ..., "status": ...} for each finished request.

print("--- Finished Involved Continuous Batching Example ---")


