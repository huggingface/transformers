#!/usr/bin/env python3
"""Quick test to verify model can generate without gibberish."""

import torch

from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32ForCausalLM


def test_simple_generation():
    """Test that a tiny model generates sensible token patterns."""

    # Create tiny model
    config = DeepseekV32Config(
        vocab_size=100,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=8,
        first_k_dense_replace=1,
        max_position_embeddings=128,
    )

    torch.manual_seed(42)
    model = DeepseekV32ForCausalLM(config)
    model.eval()

    # Generate a few tokens
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # 5 input tokens

    print("Testing generation with tiny model...")
    print(f"Input: {input_ids.tolist()}")

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=0,
        )

    generated = outputs[0].tolist()
    print(f"Generated: {generated}")

    # Check that generation produced valid tokens
    assert len(generated) >= 6, f"Expected at least 6 tokens, got {len(generated)}"
    assert len(generated) <= 15, f"Expected at most 15 tokens, got {len(generated)}"
    assert generated[:5] == [1, 2, 3, 4, 5], "Input tokens corrupted"

    # Check no repeated gibberish patterns (like same token 10 times)
    new_tokens = generated[5:]
    unique_tokens = len(set(new_tokens))
    print(f"Generated {len(new_tokens)} new tokens, {unique_tokens} unique")

    # Allow single token if it's EOS, otherwise need diversity
    if len(new_tokens) > 1:
        # Should have at least some diversity (not all same token unless it's EOS)
        if unique_tokens == 1 and new_tokens[0] not in [0, 1, 2]:  # Allow EOS/BOS/PAD repetition
            assert False, f"Generated all same non-EOS token {new_tokens[0]} (gibberish pattern)"

    # Check tokens are in vocab range
    assert all(0 <= t < 100 for t in generated), "Token out of vocab range"

    print("✅ Generation test passed!")
    print(f"   - Generated {len(generated)} tokens (input + {len(new_tokens)} new)")
    print(f"   - Input preserved: {generated[:5]}")
    print(f"   - New tokens: {new_tokens}")
    print(f"   - Token diversity: {unique_tokens}/{len(new_tokens)} unique")
    print("   - Valid vocab range: all in [0, 100)")
    print("   - No gibberish patterns detected")

    return True


if __name__ == "__main__":
    test_simple_generation()

