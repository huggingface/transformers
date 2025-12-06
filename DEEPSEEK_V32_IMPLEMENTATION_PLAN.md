# DeepSeek V3.2 Implementation Plan

## Overview

This document describes the implementation plan for adding `deepseek_v32` model support to HuggingFace Transformers, based on the official DeepSeek-V3.2-Exp release.

**Key Innovation**: DeepSeek V3.2 = DeepSeek V3 + DeepSeek Sparse Attention (DSA)

## References

- **Official Repository**: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- **Technical Report**: DeepSeek_V3_2.pdf (in the repo)
- **HuggingFace Model**: https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
- **Reference Implementation**: `/tmp/DeepSeek-V3.2-Exp/inference/model.py`

## Architecture Summary

### Model Configuration (671B)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | 129280 | Vocabulary size |
| `hidden_size` | 7168 | Model dimension (`dim` in reference) |
| `intermediate_size` | 18432 | Dense MLP intermediate size (`inter_dim`) |
| `moe_intermediate_size` | 2048 | MoE expert intermediate size (`moe_inter_dim`) |
| `num_hidden_layers` | 61 | Number of transformer layers (`n_layers`) |
| `first_k_dense_replace` | 3 | First k layers use dense MLP (`n_dense_layers`) |
| `num_attention_heads` | 128 | Number of attention heads (`n_heads`) |
| `n_routed_experts` | 256 | Number of routed experts |
| `n_shared_experts` | 1 | Number of shared experts |
| `num_experts_per_tok` | 8 | Activated experts per token (`n_activated_experts`) |
| `n_group` | 8 | Expert groups (`n_expert_groups`) |
| `topk_group` | 4 | Groups selected per token (`n_limited_groups`) |
| `routed_scaling_factor` | 2.5 | MoE routing scale (`route_scale`) |
| `scoring_func` | "sigmoid" | MoE scoring function (`score_func`) |
| `q_lora_rank` | 1536 | Query LoRA rank |
| `kv_lora_rank` | 512 | KV LoRA rank |
| `qk_nope_head_dim` | 128 | QK dimension without RoPE |
| `qk_rope_head_dim` | 64 | QK dimension with RoPE |
| `v_head_dim` | 128 | Value head dimension |
| **`index_n_heads`** | 64 | **NEW: Indexer heads** |
| **`index_head_dim`** | 128 | **NEW: Indexer head dimension** |
| **`index_topk`** | 2048 | **NEW: Top-k tokens for sparse attention** |

### Key Components

1. **MLA (Multi-Head Latent Attention)** - Same as V3
   - LoRA-compressed Q/KV projections
   - Split head dims: `qk_nope_head_dim` + `qk_rope_head_dim`
   - **Interleaved RoPE** layout

2. **Lightning Indexer** - **NEW in V3.2**
   - Computes index scores for sparse token selection
   - **Non-interleaved RoPE** layout (critical difference!)
   - Uses Hadamard transform for activation rotation
   - Learnable parameters: `wq_b`, `wk`, `k_norm`, `weights_proj`

3. **MoE** - Same as V3
   - Sigmoid scoring with group routing
   - Shared experts always active

4. **YaRN RoPE** - Same as V3
   - Extended context support

## Training Strategy (from Technical Report)

DeepSeek trains the sparse attention in **two stages**:

### Stage 1: Dense Warm-up (Indexer Only)
- **Duration**: 1000 steps, 2.1B tokens
- **Learning rate**: 1e-3
- **What's trained**: Only the Lightning Indexer
- **What's frozen**: All other model parameters
- **Attention**: Dense (full attention)
- **Objective**: KL-divergence loss to align indexer with main attention distribution

```
L_I = sum_t DKL(p_t,: || Softmax(I_t,:))
```

Where `p_t,:` is the L1-normalized sum of main attention scores across heads.

### Stage 2: Sparse Training (Full Model)
- **Duration**: 15000 steps, 943.7B tokens
- **Learning rate**: 7.3e-6
- **What's trained**: All parameters (main model + indexer)
- **Attention**: Sparse (top-k = 2048)
- **Key detail**: Indexer input is **detached** from computational graph
  - Indexer optimized only via L_I (KL loss)
  - Main model optimized only via language modeling loss

```
L_I = sum_t DKL(p_t,S_t || Softmax(I_t,S_t))
```

Where `S_t` is the set of selected top-k tokens.

## Implementation Approach

### Strategy: Modular Extension of DeepSeek V3

We extend `deepseek_v3` with minimal changes, adding only the Indexer and sparse attention logic.

### Files to Create

```
src/transformers/models/deepseek_v32/
├── __init__.py
├── configuration_deepseek_v32.py
├── modular_deepseek_v32.py          # Source of truth
└── modeling_deepseek_v32.py         # Auto-generated
```

### Files to Modify

1. `src/transformers/models/__init__.py` - Add import
2. `src/transformers/models/auto/configuration_auto.py` - Register config
3. `src/transformers/models/auto/modeling_auto.py` - Register models

### New Classes

| Class | Extends | Description |
|-------|---------|-------------|
| `DeepseekV32Config` | `DeepseekV3Config` | Adds indexer config params |
| `DeepseekV32Indexer` | `nn.Module` | Lightning Indexer implementation |
| `DeepseekV32Attention` | `DeepseekV3Attention` | Adds indexer + sparse attention |
| `DeepseekV32DecoderLayer` | `DeepseekV3DecoderLayer` | Uses new attention |
| `DeepseekV32Model` | `DeepseekV3Model` | Uses new decoder layers |
| `DeepseekV32ForCausalLM` | `DeepseekV3ForCausalLM` | Main model class |

### Hadamard Transform Strategy

**Option B: Optional with Pure PyTorch Fallback**

```python
try:
    from fast_hadamard_transform import hadamard_transform
    HAS_FAST_HADAMARD = True
except ImportError:
    HAS_FAST_HADAMARD = False
    logger.warning(
        "fast-hadamard-transform not installed. Using slower PyTorch fallback. "
        "Install with: pip install fast-hadamard-transform"
    )

def hadamard_transform_fallback(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Pure PyTorch Hadamard transform (slower but CPU-compatible)."""
    dim = x.shape[-1]
    # Pad to power of 2 if needed
    if dim & (dim - 1) != 0:
        next_pow2 = 1 << (dim - 1).bit_length()
        x = F.pad(x, (0, next_pow2 - dim))
        dim = next_pow2

    # Fast Walsh-Hadamard Transform
    h = 1
    while h < dim:
        for i in range(0, dim, h * 2):
            for j in range(i, i + h):
                a = x[..., j]
                b = x[..., j + h]
                x[..., j] = a + b
                x[..., j + h] = a - b
        h *= 2

    return x * scale

def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard transform for activation rotation."""
    hidden_size = x.size(-1)
    if HAS_FAST_HADAMARD:
        return hadamard_transform(x.contiguous(), scale=hidden_size ** -0.5)
    else:
        return hadamard_transform_fallback(x.clone(), scale=hidden_size ** -0.5)
```

**Note on fallback limitations**: The pure PyTorch fallback will be significantly slower (10-100x) than the CUDA version. For production training, `fast-hadamard-transform` should be installed.

### Training Support

All components are **fully trainable** to support:
- Full fine-tuning
- LoRA/adapter on any component
- Freezing specific components (like the indexer during warm-up)

Key implementation details for training:

1. **No `@torch.no_grad()`** - All forward passes support gradients
2. **Detachable indexer input** - Config flag `detach_indexer_input` (default: False for inference, can be set True for Stage 2 training)
3. **Indexer loss computation** - Helper method to compute KL divergence loss for indexer training
4. **Sparse attention toggle** - Config flag `use_sparse_attention` to enable/disable

### RoPE Layout Critical Note

From the technical report update:

> "The input tensor to RoPE in the indexer module requires a **non-interleaved** layout, whereas RoPE in the MLA module expects an **interleaved** layout."

Implementation:
```python
# In MLA (main attention) - interleaved RoPE
q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=True)

# In Indexer - non-interleaved RoPE
q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=False)
k_pe = apply_rotary_emb(k_pe, freqs_cis, interleaved=False)
```

## Sparse Attention Training Recommendation

Based on the technical report, here's the recommended training approach:

### For Fine-tuning from V3.2 Checkpoint
- Use sparse attention (same as inference)
- All parameters trainable
- Optionally detach indexer input for separate optimization

### For Training from Scratch or V3 Checkpoint
Follow the two-stage approach from the paper:

**Stage 1: Dense Warm-up**
```python
# Freeze all except indexer
for name, param in model.named_parameters():
    if "indexer" not in name:
        param.requires_grad = False

# Use dense attention
model.config.use_sparse_attention = False

# Train with KL loss on indexer
```

**Stage 2: Sparse Training**
```python
# Unfreeze all parameters
for param in model.parameters():
    param.requires_grad = True

# Enable sparse attention
model.config.use_sparse_attention = True

# Detach indexer input for separate optimization
model.config.detach_indexer_input = True

# Train with:
# - Language modeling loss for main model
# - KL loss for indexer (computed separately)
```

### Configuration Flags for Training

| Flag | Default | Description |
|------|---------|-------------|
| `use_sparse_attention` | True | Enable/disable sparse attention |
| `detach_indexer_input` | False | Detach indexer input from main model graph |
| `index_topk` | 2048 | Number of tokens to select |

## Testing Plan

1. **Unit tests**: Test each component (Indexer, Attention, etc.)
2. **Integration test**: Load tiny model, run forward pass
3. **Numerical equivalence**: Compare with reference implementation
4. **Gradient flow**: Verify gradients flow through all components

## Timeline

1. Configuration class (~50 lines)
2. Modular implementation (~400 lines)
3. Auto-generation of modeling file
4. Registration in auto mappings
5. Basic tests

## Open Questions / Future Work

1. **FP8 support**: The reference uses FP8 quantization. This could be added later as an optimization.
2. **FlashAttention integration**: Sparse attention with FlashAttention kernels
3. **Gradient checkpointing**: For memory-efficient training
