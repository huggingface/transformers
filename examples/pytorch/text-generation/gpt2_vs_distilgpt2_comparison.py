import time
from transformers import pipeline

# Define models to compare
models_to_compare = ["distilgpt2", "gpt2"]

# Define prompt
prompt = "The future of AI is"

# Loop over models and run text generation
for model_name in models_to_compare:
    print(f"\n--- Running with {model_name} ---")
    generator = pipeline("text-generation", model=model_name)

    # Measure execution time
    start_time = time.time()
    output = generator(prompt, max_length=30, num_return_sequences=1)
    elapsed_time = time.time() - start_time

    # Print output and timing
    print(f"Output: {output[0]['generated_text']}")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
