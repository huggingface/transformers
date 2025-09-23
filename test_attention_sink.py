#!/usr/bin/env python3
import sys

import torch


# Add transformers to path
sys.path.insert(0, "/mnt/vast/home/samuel.barry/workspace/transformers/src")

from transformers.integrations.flex_attention import flex_attention_forward as flex_attention_new
from transformers.integrations.flex_attention_old import flex_attention_forward as flex_attention_old


def test_attention_sinks():
    """Test attention sinks with old vs new implementation."""

    # Setup test data - head_dim must be at least 16 for flex_attention
    batch_size, num_heads, seq_len, head_dim = (
        1,
        2,
        4,
        16,
    )
    device = torch.device("cuda")

    dtype = torch.float32

    # Create test tensors
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    # Attention sinks
    s_aux = torch.randn(num_heads, device=device, dtype=dtype)

    # Mock module
    class MockModule:
        def __init__(self):
            self.training = False

    module = MockModule()

    # Test parameters
    test_params = {
        "attention_mask": None,
        "softcap": None,
        "score_mask": None,
        "head_mask": None,
        "s_aux": s_aux,
        "causal": False,
        "block_mask": None,
    }

    print("Testing attention sinks implementation...")
    print(f"Input shapes: query {query.shape}, key {key.shape}, value {value.shape}")
    print(f"s_aux shape: {s_aux.shape}")

    # Test old implementation
    print("\n--- Old Implementation ---")
    try:
        old_result = flex_attention_old(module, query, key, value, **test_params)
        print("Status: SUCCESS")
        print(f"Output: {old_result}")
    except Exception as e:
        # only print the first 2 lines of the error:
        e = str(e).split("\n")[:2]
        print(f"Status: FAILED - {e}")

    # Test new implementation
    print("\n--- New Implementation ---")
    try:
        print("Before calling flex_attention_new...")
        new_result = flex_attention_new(module, query, key, value, **test_params)
        print("After calling flex_attention_new...")
        print("Status: SUCCESS")
        print(f"Output: {new_result}")
    except Exception as e:
        e = str(e).split("\n")[:2]
        print(f"Status: FAILED - {e}")


if __name__ == "__main__":
    test_attention_sinks()
