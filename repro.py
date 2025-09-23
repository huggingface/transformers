#!/usr/bin/env python3
"""
Reproduction script for transformers tied tensor issue.
"""

import torch
import torch.nn as nn


def reproduce_meta_tensor_error():
    """Reproduce the meta tensor error with tied weights."""

    # Create a simple model with tied weights
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            # embed_tokens on meta device (simulates device_map behavior)
            with torch.device("meta"):
                self.embed_tokens = nn.Embedding(1000, 128)

            # lm_head on CPU (simulates loaded from checkpoint)
            self.lm_head = nn.Linear(128, 1000, bias=False)

        def tie_weights(self):
            # This is what causes the problem
            self.lm_head.weight = self.embed_tokens.weight

    model = SimpleModel()

    print(f"embed_tokens device: {model.embed_tokens.weight.device}")
    print(f"lm_head device: {model.lm_head.weight.device}")

    print("Tying weights...")
    model.tie_weights()

    print(f"After tying - lm_head device: {model.lm_head.weight.device}")
    print("Attempting to move to CPU...")

    try:
        model = model.to("cpu")
        print("Success - no error")
    except Exception as e:
        print(f"Error: {e}")
        return True

    return False


def test_with_transformers():
    """Test with actual transformers model."""
    try:
        from transformers import AutoModelForCausalLM

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

        embed_device = model.model.decoder.embed_tokens.weight.device
        lm_head_device = model.lm_head.weight.device

        print(f"embed_tokens device: {embed_device}")
        print(f"lm_head device: {lm_head_device}")

        # Check if tied
        tied = model.model.decoder.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
        print(f"Weights tied: {tied}")

        model = model.to("cpu")
        print("OPT model works fine")

    except Exception as e:
        print(f"Error with transformers: {e}")


if __name__ == "__main__":
    print("Reproducing tied tensor meta device error")
    print("-" * 40)

    error_reproduced = reproduce_meta_tensor_error()

    print("\n" + "-" * 40)
    test_with_transformers()

    print("\n" + "-" * 40)
    if error_reproduced:
        print("Error successfully reproduced!")
    else:
        print("No error reproduced.")
