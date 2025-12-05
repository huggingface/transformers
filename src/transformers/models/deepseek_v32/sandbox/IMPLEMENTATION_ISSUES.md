# DeepSeek V3.2 HuggingFace Implementation Issues

This document details the discrepancies between the HuggingFace Transformers implementation of DeepSeek V3.2 (`modeling_deepseek_v32.py`) and the reference implementation (`reference/model.py`). These issues cause the model to generate gibberish output.

## Debugging Status

| Issue | Status | Notes |
|-------|--------|-------|
| Weight name mismatches | ✅ RULED OUT | Checkpoint uses HF naming convention |
| Missing Hadamard transform | ✅ FIXED | Now raises ImportError if not installed |
| Indexer during decode | ✅ FIXED | Now applied for all seq_len, with proper caching |
| Config mismatch | ⚠️ TO CHECK | Need to compare configs |
| Numerical issues | ⚠️ TO CHECK | Run debug_forward_pass.py |

## Table of Contents

1. [~~Critical: Weight Name Mismatches~~](#critical-weight-name-mismatches) - RULED OUT
2. [~~Critical: Missing Sparse Attention During Decode~~](#critical-missing-sparse-attention-during-decode) - FIXED
3. [~~Critical: Hadamard Transform Silently Disabled~~](#critical-hadamard-transform-silently-disabled) - FIXED
4. [Medium: Indexer Weight Dtype Mismatch](#medium-indexer-weight-dtype-mismatch)
5. [Low: Minor Implementation Differences](#low-minor-implementation-differences)

---

## Critical: Weight Name Mismatches

**Severity:** Critical
**Impact:** Weights fail to load correctly, resulting in random/uninitialized weights and gibberish output.

The HF implementation uses different parameter names than the reference implementation. When loading a checkpoint saved with reference naming conventions, the weights will not map to the correct parameters.

### MLA (Multi-head Latent Attention) Layer

| Reference Name | HF Name | Shape (for V3.2 671B) |
|---------------|---------|----------------------|
| `layers.*.attn.wq_a` | `layers.*.self_attn.q_a_proj` | `[q_lora_rank, hidden_size]` |
| `layers.*.attn.q_norm` | `layers.*.self_attn.q_a_layernorm` | `[q_lora_rank]` |
| `layers.*.attn.wq_b` | `layers.*.self_attn.q_b_proj` | `[num_heads * qk_head_dim, q_lora_rank]` |
| `layers.*.attn.wkv_a` | `layers.*.self_attn.kv_a_proj_with_mqa` | `[kv_lora_rank + qk_rope_head_dim, hidden_size]` |
| `layers.*.attn.kv_norm` | `layers.*.self_attn.kv_a_layernorm` | `[kv_lora_rank]` |
| `layers.*.attn.wkv_b` | `layers.*.self_attn.kv_b_proj` | `[num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]` |
| `layers.*.attn.wo` | `layers.*.self_attn.o_proj` | `[hidden_size, num_heads * v_head_dim]` |

### Indexer (Lightning Indexer for DSA)

| Reference Name | HF Name | Shape |
|---------------|---------|-------|
| `layers.*.attn.indexer.wq_b` | `layers.*.self_attn.indexer.wq_b` | Same ✓ |
| `layers.*.attn.indexer.wk` | `layers.*.self_attn.indexer.wk` | Same ✓ |
| `layers.*.attn.indexer.k_norm` | `layers.*.self_attn.indexer.k_norm` | Same ✓ |
| `layers.*.attn.indexer.weights_proj` | `layers.*.self_attn.indexer.weights_proj` | Same ✓ |

Note: Indexer weights have matching names, but the parent path differs (`attn` vs `self_attn`).

### MLP (Dense Layers, layer_idx < first_k_dense_replace)

| Reference Name | HF Name | Shape |
|---------------|---------|-------|
| `layers.*.ffn.w1` | `layers.*.mlp.gate_proj` | `[intermediate_size, hidden_size]` |
| `layers.*.ffn.w2` | `layers.*.mlp.down_proj` | `[hidden_size, intermediate_size]` |
| `layers.*.ffn.w3` | `layers.*.mlp.up_proj` | `[intermediate_size, hidden_size]` |

### MoE (Mixture of Experts, layer_idx >= first_k_dense_replace)

| Reference Name | HF Name | Shape |
|---------------|---------|-------|
| `layers.*.ffn.gate.weight` | `layers.*.mlp.gate.weight` | Same ✓ |
| `layers.*.ffn.gate.bias` | `layers.*.mlp.gate.e_score_correction_bias` | `[n_routed_experts]` |
| `layers.*.ffn.experts.*.w1` | `layers.*.mlp.experts.*.gate_proj` | `[moe_intermediate_size, hidden_size]` |
| `layers.*.ffn.experts.*.w2` | `layers.*.mlp.experts.*.down_proj` | `[hidden_size, moe_intermediate_size]` |
| `layers.*.ffn.experts.*.w3` | `layers.*.mlp.experts.*.up_proj` | `[moe_intermediate_size, hidden_size]` |
| `layers.*.ffn.shared_experts.w1` | `layers.*.mlp.shared_experts.gate_proj` | `[n_shared * moe_inter, hidden_size]` |
| `layers.*.ffn.shared_experts.w2` | `layers.*.mlp.shared_experts.down_proj` | `[hidden_size, n_shared * moe_inter]` |
| `layers.*.ffn.shared_experts.w3` | `layers.*.mlp.shared_experts.up_proj` | `[n_shared * moe_inter, hidden_size]` |

### Layer Norms

| Reference Name | HF Name |
|---------------|---------|
| `layers.*.attn_norm` | `layers.*.input_layernorm` |
| `layers.*.ffn_norm` | `layers.*.post_attention_layernorm` |

### Model-Level

| Reference Name | HF Name |
|---------------|---------|
| `embed` | `model.embed_tokens` |
| `norm` | `model.norm` |
| `head` | `lm_head` |

### Fix Required

Add a weight name mapping in the model's `_load_pretrained_model` method or create a checkpoint conversion script:

```python
WEIGHT_NAME_MAPPING = {
    "layers.{}.attn.wq_a": "layers.{}.self_attn.q_a_proj",
    "layers.{}.attn.q_norm": "layers.{}.self_attn.q_a_layernorm",
    "layers.{}.attn.wq_b": "layers.{}.self_attn.q_b_proj",
    "layers.{}.attn.wkv_a": "layers.{}.self_attn.kv_a_proj_with_mqa",
    "layers.{}.attn.kv_norm": "layers.{}.self_attn.kv_a_layernorm",
    "layers.{}.attn.wkv_b": "layers.{}.self_attn.kv_b_proj",
    "layers.{}.attn.wo": "layers.{}.self_attn.o_proj",
    "layers.{}.attn.indexer": "layers.{}.self_attn.indexer",
    "layers.{}.attn_norm": "layers.{}.input_layernorm",
    "layers.{}.ffn_norm": "layers.{}.post_attention_layernorm",
    "layers.{}.ffn.w1": "layers.{}.mlp.gate_proj",
    "layers.{}.ffn.w2": "layers.{}.mlp.down_proj",
    "layers.{}.ffn.w3": "layers.{}.mlp.up_proj",
    "layers.{}.ffn.gate.bias": "layers.{}.mlp.gate.e_score_correction_bias",
    "layers.{}.ffn.experts.{}.w1": "layers.{}.mlp.experts.{}.gate_proj",
    "layers.{}.ffn.experts.{}.w2": "layers.{}.mlp.experts.{}.down_proj",
    "layers.{}.ffn.experts.{}.w3": "layers.{}.mlp.experts.{}.up_proj",
    "layers.{}.ffn.shared_experts.w1": "layers.{}.mlp.shared_experts.gate_proj",
    "layers.{}.ffn.shared_experts.w2": "layers.{}.mlp.shared_experts.down_proj",
    "layers.{}.ffn.shared_experts.w3": "layers.{}.mlp.shared_experts.up_proj",
    "embed": "model.embed_tokens",
    "head": "lm_head",
}
```

---

## Critical: Missing Sparse Attention During Decode

**Severity:** Critical
**Impact:** Different attention patterns during autoregressive generation, potentially causing degraded output quality.

### Reference Implementation (model.py:641-659)

The reference applies the Lightning Indexer during **both** prefill and decode:

```python
# Prefill (mask is not None, seqlen > 1)
if mask is not None:
    # ... full attention with indexer
    topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
    index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
    # ...

# Decode (mask is None, seqlen == 1)
else:
    # ... MQA decode with indexer
    topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
    index_mask = torch.full((bsz, 1, end_pos), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
    scores += index_mask.unsqueeze(2)
```

### HF Implementation (modular_deepseek_v32.py:964)

The HF implementation only applies the indexer during prefill:

```python
# Apply sparse attention via indexer (during prefill with full mask)
if seq_len > 1:  # <-- BUG: Skips indexer during decode!
    topk_indices, _ = self.indexer(...)
    index_mask = ...
    attn_weights = attn_weights + index_mask.unsqueeze(1)
```

### Fix Required

Apply the indexer during decode as well:

```python
# In DeepseekV32Attention.forward()
if seq_len > 1:
    # Prefill: apply indexer with causal mask
    topk_indices, _ = self.indexer(hidden_states, q_compressed, freqs_cis, attention_mask.squeeze(1))
    index_mask = torch.full((batch_size, seq_len, k_states.shape[-2]), float("-inf"), ...)
    index_mask.scatter_(-1, topk_indices, 0.0)
    attn_weights = attn_weights + index_mask.unsqueeze(1)
else:
    # Decode: also apply indexer (no causal mask needed for single token)
    topk_indices, _ = self.indexer(hidden_states, q_compressed, freqs_cis, None)
    index_mask = torch.full((batch_size, 1, k_states.shape[-2]), float("-inf"), ...)
    index_mask.scatter_(-1, topk_indices, 0.0)
    attn_weights = attn_weights + index_mask.unsqueeze(1)
```

---

## Critical: Hadamard Transform Silently Disabled

**Severity:** Critical
**Impact:** Indexer computes incorrect scores when `fast_hadamard_transform` is not installed.

### Reference Implementation (model.py:445-450)

The reference **requires** the Hadamard transform:

```python
def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)
```

### HF Implementation (modular_deepseek_v32.py:340-358)

The HF implementation silently falls back to identity:

```python
def hadamard_transform_activation(x: torch.Tensor) -> torch.Tensor:
    try:
        from fast_hadamard_transform import hadamard_transform
    except ImportError:
        return x  # <-- BUG: Silently returns identity!
    # ...
```

### Impact

The Hadamard transform is essential for the indexer to compute correct sparse attention indices. Without it:
- The Q and K representations are not properly transformed
- Index scores become meaningless
- Wrong tokens are selected for attention
- Output degrades significantly

### Fix Required

Either:

1. **Make it mandatory** and raise an error:
```python
def hadamard_transform_activation(x: torch.Tensor) -> torch.Tensor:
    try:
        from fast_hadamard_transform import hadamard_transform
    except ImportError:
        raise ImportError(
            "DeepSeek V3.2 requires the fast_hadamard_transform package for the "
            "Lightning Indexer. Install it with: pip install fast-hadamard-transform"
        )
    # ...
```

2. **Or warn loudly** at model initialization:
```python
# In DeepseekV32Indexer.__init__()
try:
    from fast_hadamard_transform import hadamard_transform
    self._has_hadamard = True
except ImportError:
    self._has_hadamard = False
    logger.warning_once(
        "fast_hadamard_transform not installed. The Lightning Indexer will not "
        "function correctly, which may significantly degrade output quality. "
        "Install with: pip install fast-hadamard-transform"
    )
```

---

## Medium: Indexer Weight Dtype Mismatch

**Severity:** Medium
**Impact:** Potential numerical differences in indexer score computation.

### Reference Implementation (model.py:467)

```python
# weights_proj in the checkpoint is stored in bf16, while the parameters
# here are stored in fp32 for convenient.
self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.float32)
```

### HF Implementation (modular_deepseek_v32.py:716)

```python
self.weights_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
# Uses model's default dtype, typically bfloat16
```

### Fix Required

Explicitly set float32 dtype:

```python
self.weights_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False, dtype=torch.float32)
```

---

## Low: Minor Implementation Differences

These differences are unlikely to cause gibberish but may result in subtle numerical differences:

### 1. KV Cache Strategy

**Reference:** Caches compressed KV (`kv_lora_rank` dimensions) and k_pe separately, then projects during decode using absorbed wkv_b projection.

**HF:** Caches fully projected K and V (`num_heads * qk_head_dim` and `num_heads * v_head_dim`), which is less memory-efficient but functionally equivalent.

### 2. RMSNorm Residual Fusion

**Reference:** Fuses residual addition inside RMSNorm for efficiency:
```python
def forward(self, x, residual=None):
    if residual is not None:
        x = residual = x.float() + residual.float()
    # ... normalize x
    return normalized_x, residual
```

**HF:** Standard pre-norm with separate residual addition (functionally equivalent).

### 3. Softmax Scale Precision

**Reference:** Uses `mul_()` for in-place multiplication:
```python
scores = torch.einsum("bshd,bthd->bsht", q, k).mul_(self.softmax_scale)
```

**HF:** Uses standard multiplication:
```python
attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.softmax_scale
```

These are functionally equivalent.

---

## Summary of Required Fixes

| Priority | Issue | Fix | Status |
|----------|-------|-----|--------|
| P0 | Weight name mismatches | Add weight name mapping for checkpoint loading | RULED OUT |
| P0 | Missing indexer during decode | Apply indexer for seq_len == 1 | ✅ FIXED |
| P0 | Silent Hadamard fallback | Raise error or warn when package missing | ✅ FIXED |
| P1 | weights_proj dtype | Use explicit float32 dtype | TO DO |

## Testing Recommendations

After fixing these issues:

1. **Weight loading test:** Verify all weights are loaded with no missing/unexpected keys
2. **Forward pass test:** Compare hidden states at each layer against reference
3. **Indexer test:** Verify top-k indices match reference implementation
4. **Generation test:** Compare generated text quality with reference model
