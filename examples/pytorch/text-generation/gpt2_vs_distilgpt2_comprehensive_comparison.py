import math
import time

import torch
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# Models to compare
models_to_compare = ["distilgpt2", "gpt2"]

# Prompts to test
prompts = [
    "The future of AI is",
    "In a world where technology",
    "The role of open source is",
]

# Load a small evaluation set for perplexity calculation
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:5]")
texts = [x["text"] for x in dataset if len(x["text"]) > 0]


# Function to calculate perplexity
def calculate_perplexity(model_name, dataset_texts):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    losses = []
    for text in dataset_texts:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    return math.exp(avg_loss)


# Main comparison loop
for model_name in models_to_compare:
    print(f"\n=== Model: {model_name} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Perplexity Evaluation
    perplexity = calculate_perplexity(model_name, texts)
    print(f"Perplexity on wikitext-2 (5 samples): {perplexity:.2f}")

    # Generation Pipeline
    generator = pipeline("text-generation", model=model_name)

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        start_time = time.time()
        output = generator(prompt, max_length=30, num_return_sequences=1)[0]["generated_text"]
        elapsed_time = time.time() - start_time

        token_count = len(tokenizer(output)["input_ids"])
        unique_tokens = len(set(tokenizer(output)["input_ids"]))

        print(f"Generated Output: {output}")
        print(f"Generation Time: {elapsed_time:.2f} seconds")
        print(f"Output Length (tokens): {token_count}")
        print(f"Unique Token Count: {unique_tokens}")
