#!/usr/bin/env python3
"""
Example: Using Flash Attention 4 (FA4) with HuggingFace Transformers

This example demonstrates how to use FA4 with transformer models for inference.

Requirements:
    - CUDA-capable GPU with SM 8.0+ (Ampere, Hopper, or Blackwell)
    - flash-attn package with CuTe DSL support
    - transformers with FA4 integration

Usage:
    python examples/flash_attention_4_example.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, is_flash_attn_4_available


def check_fa4_availability():
    """Check if FA4 is available on this system."""
    print("Checking FA4 availability...")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. FA4 requires CUDA GPU.")
        return False

    major, minor = torch.cuda.get_device_capability()
    compute_cap = major * 10 + minor

    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Compute Capability: SM {major}.{minor}")

    if compute_cap < 80:
        print(f"  ERROR: FA4 requires SM 8.0+, found SM {major}.{minor}")
        return False

    if compute_cap < 90:
        print(f"  NOTE: FA4 is optimized for SM 9.0+")
        print(f"        Your GPU will have reduced optimizations")

    fa4_available = is_flash_attn_4_available()
    print(f"  FA4 Available: {fa4_available}")

    if not fa4_available:
        print("\n  FA4 not available. Possible reasons:")
        print("  1. flash-attn package not installed")
        print("  2. flash-attn version doesn't include CuTe DSL")
        print("  3. GPU compute capability insufficient")
        print("\n  Install with: pip install flash-attn --upgrade")

    return fa4_available


def example_basic_inference():
    """Example 1: Basic inference with FA4."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Inference with FA4")
    print("=" * 70)

    model_name = "gpt2"
    print(f"\nLoading model: {model_name}")

    # Load model with explicit FA4 attention
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_4",  # Explicitly request FA4
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Verify FA4 is being used
    print(f"Attention implementation: {model.config._attn_implementation}")

    # Generate text
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"\nPrompt: {prompt}")
    print("Generating...")

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated: {generated_text}")


def example_auto_selection():
    """Example 2: Automatic FA4 selection."""
    print("\n" + "=" * 70)
    print("Example 2: Automatic FA4 Selection")
    print("=" * 70)

    model_name = "gpt2"
    print(f"\nLoading model: {model_name}")

    # Load without specifying attention implementation
    # Will auto-select best available (FA4 > FA3 > FA2 > SDPA)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
        # attn_implementation not specified - will auto-select
    )

    print(f"Auto-selected attention: {model.config._attn_implementation}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "Machine learning is revolutionizing"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"\nPrompt: {prompt}")
    print("Generating...")

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=50)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated: {generated_text}")


def example_compare_implementations():
    """Example 3: Compare FA4 vs other implementations."""
    print("\n" + "=" * 70)
    print("Example 3: Comparing Attention Implementations")
    print("=" * 70)

    model_name = "gpt2"
    prompt = "The quick brown fox"

    implementations = ["flash_attention_4", "flash_attention_2", "sdpa", "eager"]

    for impl in implementations:
        print(f"\n--- Testing {impl} ---")

        try:
            # Load model with specific implementation
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation=impl
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Measure inference time
            import time

            torch.cuda.synchronize()
            start = time.time()

            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=20)

            torch.cuda.synchronize()
            elapsed = time.time() - start

            print(f"  Implementation: {model.config._attn_implementation}")
            print(f"  Time: {elapsed:.4f}s")

        except Exception as e:
            print(f"  ERROR: {e}")


def main():
    print("\n" + "=" * 70)
    print("  Flash Attention 4 (FA4) Example")
    print("=" * 70)

    # Check FA4 availability
    if not check_fa4_availability():
        print("\nFA4 not available. Examples will use fallback implementations.")
        print("Install flash-attn to enable FA4.")

    # Run examples
    try:
        example_basic_inference()
    except Exception as e:
        print(f"\nExample 1 failed: {e}")

    try:
        example_auto_selection()
    except Exception as e:
        print(f"\nExample 2 failed: {e}")

    try:
        example_compare_implementations()
    except Exception as e:
        print(f"\nExample 3 failed: {e}")

    print("\n" + "=" * 70)
    print("  Examples Complete")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
