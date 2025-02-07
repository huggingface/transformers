import torch
import pytest
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

def test_to_4d_packed_mask_block_diagonal():
    # Define the per-sample sequence length.
    query_length = 5
    # Pack 3 sequences into one row (total_length = 15).
    num_sequences = 3
    total_length = query_length * num_sequences

    # Create a 2D attention mask of shape (batch_size, total_length) with all ones.
    # Here, 1 indicates a valid token (not padding).
    attention_mask_2d = torch.ones((1, total_length), dtype=torch.int)

    # Instantiate the converter with causal masking enabled.
    converter = AttentionMaskConverter(is_causal=True)
    key_value_length = total_length

    # Convert the 2D mask to a 4D mask.
    converted_mask = converter.to_4d(
        attention_mask_2d,
        query_length=query_length,
        dtype=torch.float32,
        key_value_length=key_value_length
    )

    # The expected shape is (batch_size, 1, total_length, total_length).
    expected_shape = (1, 1, total_length, total_length)
    assert converted_mask.shape == expected_shape, f"Expected shape {expected_shape}, got {converted_mask.shape}"

    # Define the large negative value used for masking.
    neg_inf = torch.finfo(torch.float32).min

    # Check that each block (of size query_length x query_length) has proper causal masking.
    for block in range(num_sequences):
        start = block * query_length
        end = start + query_length
        # Extract the rows corresponding to the current block.
        block_rows = converted_mask[0, 0, start:end, :]
        # Positions outside the block (to the left and right) should be masked.
        left_side = block_rows[:, :start]
        right_side = block_rows[:, end:]
        assert torch.allclose(left_side, torch.full(left_side.shape, neg_inf, dtype=torch.float32)), f"Block {block}: Left side is not masked correctly"
        assert torch.allclose(right_side, torch.full(right_side.shape, neg_inf, dtype=torch.float32)), f"Block {block}: Right side is not masked correctly"

def test_to_4d_nonpacked_mask():
    # Test for the standard (non-packed) case where total_length equals query_length.
    query_length = 7
    batch_size = 2
    # Create a 2D attention mask of shape (batch_size, query_length) with ones.
    attention_mask_2d = torch.ones((batch_size, query_length), dtype=torch.int)

    # Instantiate the converter (here causal is not required).
    converter = AttentionMaskConverter(is_causal=False)

    # Convert the mask.
    converted_mask = converter.to_4d(
        attention_mask_2d,
        query_length=query_length,
        dtype=torch.float32,
        key_value_length=query_length  # For non-causal mode, key_value_length isn't used.
    )

    # Expected shape: (batch_size, 1, query_length, query_length)
    expected_shape = (batch_size, 1, query_length, query_length)
    assert converted_mask.shape == expected_shape, f"Expected shape {expected_shape}, got {converted_mask.shape}"
    
    # In this case, since the mask is all ones (and _expand_mask inverts them),
    # the output should be filled with the large negative value.
    neg_inf = torch.finfo(torch.float32).min
    expected_mask = torch.full(expected_shape, neg_inf, dtype=torch.float32)
    assert torch.allclose(converted_mask, expected_mask), "Non-packed mask conversion did not produce the expected result"
