#!/usr/bin/env python3
"""
Standalone test script for DeepSeek V3.2 implementation.
Tests forward pass, backward pass, and loss decreasing.

Usage:
    python scripts/test_deepseek_v32.py [--device cuda|cpu] [--dtype float32|float16|bfloat16]
"""

import argparse
import sys
import os

# Add src to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch


def get_tiny_config():
    """Get a tiny config for fast testing."""
    from transformers import DeepseekV32Config

    return DeepseekV32Config(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=2.5,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=16,
        v_head_dim=16,
        qk_nope_head_dim=16,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        max_position_embeddings=128,
        # V3.2 specific
        index_n_heads=4,
        index_head_dim=32,
        index_topk=8,
        use_sparse_attention=True,
    )


def get_small_config():
    """Get a small but more realistic config for testing."""
    from transformers import DeepseekV32Config

    return DeepseekV32Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=64,
        q_lora_rank=128,
        qk_rope_head_dim=32,
        v_head_dim=32,
        qk_nope_head_dim=32,
        n_group=4,
        topk_group=2,
        num_experts_per_tok=4,
        first_k_dense_replace=1,
        max_position_embeddings=512,
        # V3.2 specific
        index_n_heads=8,
        index_head_dim=64,
        index_topk=64,
        use_sparse_attention=True,
    )


def test_forward_pass(model, device, dtype):
    """Test forward pass."""
    print("\n" + "=" * 60)
    print("TEST: Forward Pass")
    print("=" * 60)

    model.eval()
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            outputs = model(input_ids)

    logits = outputs.logits

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Output dtype: {logits.dtype}")

    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()

    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    print(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")

    if has_nan or has_inf:
        print("  FAILED: Output contains NaN or Inf")
        return False

    print("  PASSED")
    return True


def test_backward_pass(model, device, dtype):
    """Test backward pass."""
    print("\n" + "=" * 60)
    print("TEST: Backward Pass")
    print("=" * 60)

    model.train()
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    model.zero_grad()

    with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

    print(f"  Loss: {loss.item():.4f}")

    has_nan = torch.isnan(loss).item()
    has_inf = torch.isinf(loss).item()

    print(f"  Loss has NaN: {has_nan}")
    print(f"  Loss has Inf: {has_inf}")

    if has_nan or has_inf:
        print("  FAILED: Loss contains NaN or Inf")
        return False

    # Backward pass
    loss.backward()

    # Check gradients
    total_params = 0
    params_with_grad = 0
    indexer_params_with_grad = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is not None:
                grad_norm = param.grad.abs().sum().item()
                if grad_norm > 0:
                    params_with_grad += 1
                    if "indexer" in name:
                        indexer_params_with_grad += 1

    print(f"  Total trainable params: {total_params}")
    print(f"  Params with non-zero grad: {params_with_grad}")
    print(f"  Indexer params with grad: {indexer_params_with_grad}")

    if params_with_grad == 0:
        print("  FAILED: No gradients computed")
        return False

    print("  PASSED")
    return True


def test_loss_decreases(model, device, dtype, num_steps=20):
    """Test that loss decreases over training."""
    print("\n" + "=" * 60)
    print(f"TEST: Loss Decreases Over {num_steps} Steps")
    print("=" * 60)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch_size, seq_len = 4, 64
    torch.manual_seed(42)
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if step % 5 == 0 or step == num_steps - 1:
            print(f"  Step {step:3d}: loss = {loss.item():.4f}")

    # Check if loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]
    decrease_pct = (initial_loss - final_loss) / initial_loss * 100

    print(f"\n  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Decrease: {decrease_pct:.1f}%")

    if final_loss >= initial_loss:
        print("  FAILED: Loss did not decrease")
        return False

    print("  PASSED")
    return True


def test_sparse_vs_dense(model, device, dtype):
    """Test both sparse and dense attention modes."""
    print("\n" + "=" * 60)
    print("TEST: Sparse vs Dense Attention Modes")
    print("=" * 60)

    model.eval()
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    # Test with sparse attention (default)
    model.config.use_sparse_attention = True
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            outputs_sparse = model(input_ids)

    print(f"  Sparse attention - logits range: [{outputs_sparse.logits.min().item():.4f}, {outputs_sparse.logits.max().item():.4f}]")
    has_nan_sparse = torch.isnan(outputs_sparse.logits).any().item()

    # Test with dense attention
    model.config.use_sparse_attention = False
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            outputs_dense = model(input_ids)

    print(f"  Dense attention - logits range: [{outputs_dense.logits.min().item():.4f}, {outputs_dense.logits.max().item():.4f}]")
    has_nan_dense = torch.isnan(outputs_dense.logits).any().item()

    # Restore sparse attention
    model.config.use_sparse_attention = True

    if has_nan_sparse or has_nan_dense:
        print("  FAILED: NaN in outputs")
        return False

    print("  PASSED")
    return True


def test_indexer_scores_output(model, device, dtype):
    """Test that indexer scores are returned when requested."""
    print("\n" + "=" * 60)
    print("TEST: Indexer Scores Output")
    print("=" * 60)

    model.eval()
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    # Ensure sparse attention is enabled
    model.config.use_sparse_attention = True

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            outputs = model(input_ids, output_indexer_scores=True)

    if outputs.indexer_scores is None:
        print("  FAILED: indexer_scores is None")
        return False

    num_layers = len(outputs.indexer_scores)
    print(f"  Number of layers with scores: {num_layers}")
    print(f"  Expected layers: {model.config.num_hidden_layers}")

    if num_layers != model.config.num_hidden_layers:
        print("  FAILED: Wrong number of layers")
        return False

    # Check shape of first layer
    scores_shape = outputs.indexer_scores[0].shape
    expected_shape = (batch_size, seq_len, seq_len)
    print(f"  Scores shape: {scores_shape}")
    print(f"  Expected shape: {expected_shape}")

    if scores_shape != expected_shape:
        print("  FAILED: Wrong scores shape")
        return False

    # Check for NaN
    has_nan = any(torch.isnan(s).any().item() for s in outputs.indexer_scores)
    if has_nan:
        print("  FAILED: Scores contain NaN")
        return False

    print("  PASSED")
    return True


def test_kl_target_output(model, device, dtype):
    """Test that KL target is computed correctly."""
    print("\n" + "=" * 60)
    print("TEST: KL Target Output")
    print("=" * 60)

    model.eval()
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    # Ensure sparse attention is enabled
    model.config.use_sparse_attention = True

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            outputs = model(
                input_ids,
                output_indexer_scores=True,
                output_indexer_kl_target=True,
            )

    if outputs.indexer_kl_targets is None:
        print("  FAILED: indexer_kl_targets is None")
        return False

    num_layers = len(outputs.indexer_kl_targets)
    print(f"  Number of layers with KL targets: {num_layers}")

    # Check shape
    target_shape = outputs.indexer_kl_targets[0].shape
    expected_shape = (batch_size, seq_len, seq_len)
    print(f"  Target shape: {target_shape}")

    if target_shape != expected_shape:
        print("  FAILED: Wrong target shape")
        return False

    # Check L1-normalization (should sum to 1 along last dim)
    target = outputs.indexer_kl_targets[0].float()
    row_sums = target.sum(dim=-1)
    expected_sums = torch.ones_like(row_sums)

    is_normalized = torch.allclose(row_sums, expected_sums, atol=1e-4)
    print(f"  Is L1-normalized: {is_normalized}")
    print(f"  Row sums range: [{row_sums.min().item():.6f}, {row_sums.max().item():.6f}]")

    if not is_normalized:
        print("  FAILED: Target not L1-normalized")
        return False

    print("  PASSED")
    return True


def test_kl_loss_computation(model, device, dtype):
    """Test KL loss can be computed and backpropagated."""
    print("\n" + "=" * 60)
    print("TEST: KL Loss Computation and Gradients")
    print("=" * 60)

    from transformers.models.deepseek_v32.modeling_deepseek_v32 import compute_indexer_kl_loss

    model.train()

    # Freeze non-indexer params
    for name, param in model.named_parameters():
        if "indexer" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    # Ensure sparse attention is enabled
    model.config.use_sparse_attention = True

    model.zero_grad()

    with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
        outputs = model(
            input_ids,
            output_indexer_scores=True,
            output_indexer_kl_target=True,
        )

        # Compute KL loss
        kl_loss = compute_indexer_kl_loss(
            outputs.indexer_scores,
            outputs.indexer_kl_targets,
        )

    print(f"  KL loss value: {kl_loss.item():.6f}")

    has_nan = torch.isnan(kl_loss).item()
    has_inf = torch.isinf(kl_loss).item()

    if has_nan or has_inf:
        print("  FAILED: KL loss contains NaN or Inf")
        return False

    if not kl_loss.requires_grad:
        print("  FAILED: KL loss doesn't require grad")
        return False

    # Backward pass
    kl_loss.backward()

    # Check indexer has gradients
    indexer_grads = 0
    for name, param in model.named_parameters():
        if "indexer" in name and param.requires_grad:
            if param.grad is not None and param.grad.abs().sum() > 0:
                indexer_grads += 1

    print(f"  Indexer params with non-zero gradients: {indexer_grads}")

    if indexer_grads == 0:
        print("  FAILED: No gradients in indexer params")
        return False

    # Restore all params to trainable
    for param in model.parameters():
        param.requires_grad = True

    print("  PASSED")
    return True


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser(description="Test DeepSeek V3.2 implementation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--config", type=str, default="tiny", choices=["tiny", "small"])
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("DeepSeek V3.2 Implementation Test")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Config: {args.config}")

    # Import model
    from transformers import DeepseekV32ForCausalLM

    # Create config and model
    if args.config == "tiny":
        config = get_tiny_config()
    else:
        config = get_small_config()

    print(f"\nModel Config:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  n_routed_experts: {config.n_routed_experts}")
    print(f"  index_n_heads: {config.index_n_heads}")
    print(f"  index_topk: {config.index_topk}")

    print("\nCreating model...")
    model = DeepseekV32ForCausalLM(config)

    # For bfloat16/float16, move model to that dtype
    if dtype != torch.float32:
        model = model.to(dtype)

    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Run tests
    results = {}

    results["forward"] = test_forward_pass(model, device, dtype)
    results["backward"] = test_backward_pass(model, device, dtype)
    results["loss_decreases"] = test_loss_decreases(model, device, dtype)
    results["sparse_vs_dense"] = test_sparse_vs_dense(model, device, dtype)
    results["indexer_scores_output"] = test_indexer_scores_output(model, device, dtype)
    results["kl_target_output"] = test_kl_target_output(model, device, dtype)
    results["kl_loss_computation"] = test_kl_loss_computation(model, device, dtype)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
