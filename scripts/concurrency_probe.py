"""Concurrency probe for strict overlay functionality."""

import concurrent.futures
import os
import sys
from typing import Any


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from assist_strict.assisted import assisted_generate_strict
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup() -> tuple[Any, Any]:
    """Initialize tokenizer and prepare test inputs for concurrency testing.

    Returns:
        Tuple of (tokenizer, tokenized_inputs) for CPU-friendly testing.
    """
    # Use small models suitable for CPU testing
    model_name = "microsoft/DialoGPT-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare test input and move to CPU
    text = "Hello world"
    tokenized_inputs = tokenizer(text, return_tensors="pt", padding=True)
    # Move all tensor values to CPU
    tokenized_inputs = {k: v.to("cpu") for k, v in tokenized_inputs.items()}

    return tokenizer, tokenized_inputs


def worker(model: Any, assistant_model: Any, tokenized_inputs: Any, n: int) -> int:
    """Execute assisted_generate_strict and return post-call num_assistant_tokens.

    Args:
        model: The primary model for generation.
        assistant_model: The assistant model for generation.
        tokenized_inputs: Pre-tokenized inputs on CPU.
        n: The num_assistant_tokens value to use for this worker.

    Returns:
        The assistant model's post-call num_assistant_tokens (should equal library default).
    """
    # Ensure input_ids are on CPU before passing to assisted_generate_strict
    input_ids_cpu = tokenized_inputs["input_ids"].to("cpu")

    # Execute assisted generation with specified num_assistant_tokens
    assisted_generate_strict(
        model=model,
        inputs=input_ids_cpu,
        assistant_model=assistant_model,
        num_assistant_tokens=n,
        max_new_tokens=5,  # Keep generation short
        do_sample=False,
        pad_token_id=input_ids_cpu[0, 0].item()  # Use first token as pad
    )

    # Return post-call num_assistant_tokens to verify restoration
    return getattr(assistant_model.generation_config, 'num_assistant_tokens', None)


def main() -> None:
    """Run concurrency probe with multiple workers and print post-call values."""
    print("Starting concurrency probe...")

    # Load models once with fully materialized weights on CPU
    model_name = "microsoft/DialoGPT-small"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,
        dtype="float32",
        low_cpu_mem_usage=False,
        _fast_init=False
    )
    assistant_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,
        dtype="float32",
        low_cpu_mem_usage=False,
        _fast_init=False
    )

    # Get tokenizer and inputs
    tokenizer, tokenized_inputs = setup()

    # Test values for different workers
    test_values = [1, 3, 5, 7]

    # Run workers concurrently, passing shared models and inputs
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, model, assistant_model, tokenized_inputs, n) for n in test_values]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Print collected post-call values for verification
    print(f"Post-call values (should be all defaults): {results}")
    print("Concurrency probe completed.")


if __name__ == "__main__":
    main()
