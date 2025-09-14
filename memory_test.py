"""
Test script to evaluate memory efficiency of CPU logits offloading.
"""

import gc
import time

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    return torch.cuda.memory_allocated() / 1024 / 1024


def clear_memory():
    """Clear GPU memory."""
    torch.cuda.empty_cache()
    gc.collect()


def test_memory_scenario():
    """Test memory efficiency of CPU offloading with gpt2-large."""

    model_name = "gpt2-large"
    max_new_tokens = 1000
    prompt_text = "This essay will start with a detailed and verbose introduction to deep learning. To start with,"

    print(f"Testing model: {model_name}")
    print(f"Tokens to generate: {max_new_tokens}")
    print(f"Device: {torch.cuda.get_device_name()}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).cuda()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}


    # Test WITHOUT CPU offloading
    clear_memory()
    print("\nTesting without CPU offloading...")

    initial_memory = get_gpu_memory_mb()
    start_time = time.time()

    with torch.no_grad():
        outputs_gpu = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            output_logits=True,
            return_dict_in_generate=True,
            offload_logits_to_cpu=False,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    gpu_time = time.time() - start_time
    gpu_peak = get_gpu_memory_mb()
    gpu_increase = gpu_peak - initial_memory

    print(f"Initial memory: {initial_memory:.1f} MB")
    print(f"Peak memory: {gpu_peak:.1f} MB")
    print(f"Memory increase: {gpu_increase:.1f} MB")
    print(f"Generation time: {gpu_time:.2f}s")
    print(f"Tokens generated: {len(outputs_gpu.logits) if outputs_gpu.logits else 0}")

    # Store for comparison
    gpu_sequences = outputs_gpu.sequences.clone()
    del outputs_gpu
    clear_memory()

    # Test WITH CPU offloading
    print("\nTesting with CPU offloading...")

    initial_memory = get_gpu_memory_mb()
    start_time = time.time()

    with torch.no_grad():
        outputs_cpu = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            output_logits=True,
            return_dict_in_generate=True,
            offload_logits_to_cpu=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    cpu_time = time.time() - start_time
    cpu_peak = get_gpu_memory_mb()
    cpu_increase = cpu_peak - initial_memory

    print(f"Initial memory: {initial_memory:.1f} MB")
    print(f"Peak memory: {cpu_peak:.1f} MB")
    print(f"Memory increase: {cpu_increase:.1f} MB")
    print(f"Generation time: {cpu_time:.2f}s")
    print(f"Tokens generated: {len(outputs_cpu.logits) if outputs_cpu.logits else 0}")

    # Analysis
    memory_saved = gpu_increase - cpu_increase
    reduction_pct = (memory_saved / gpu_increase * 100) if gpu_increase > 0 else 0
    time_overhead = ((cpu_time - gpu_time) / gpu_time * 100) if gpu_time > 0 else 0
    sequences_match = torch.equal(gpu_sequences, outputs_cpu.sequences)

    print("\nResults:")
    print("Number of tokens generated:", outputs_cpu.sequences.shape[1])
    print(f"Memory saved: {memory_saved:.1f} MB")
    print(f"Memory reduction: {reduction_pct:.1f}%")
    print(f"Time overhead: {time_overhead:+.1f}%")
    print(f"Sequences match: {sequences_match}")

if __name__ == "__main__":
    test_memory_scenario()
