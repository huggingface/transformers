# DeepSeek V3.2 Implementation Specification

## Overview

DeepSeek V3.2 introduces **DeepSeek Sparse Attention (DSA)** - an efficient attention mechanism that reduces computational complexity from O(L²) to O(Lk) while maintaining model performance. This document provides a comprehensive specification for implementing DeepSeek V3.2 in HuggingFace Transformers.

## Reference Files

The implementation is based on the following reference files from DeepSeek:

| File | Location | Purpose |
|------|----------|---------|
| `model.py` | `deepseek_files/inference/model.py` | Complete reference implementation (~920 lines) |
| `config.json` | `deepseek_files/config.json` | Official HuggingFace-format configuration |

---

## Architecture Overview

### Model Statistics (671B Configuration)

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 7168 |
| `num_hidden_layers` | 61 |
| `num_attention_heads` | 128 |
| `vocab_size` | 129280 |
| `max_position_embeddings` | 163840 |
| Total Parameters | ~671B |

### Key Innovations

1. **Lightning Indexer** - Selects top-k tokens for sparse attention
2. **Multi-head Latent Attention (MLA)** - LoRA-compressed Q/KV projections
3. **Mixture of Experts (MoE)** - 256 routed experts with sigmoid scoring
4. **YaRN Scaling** - Extended context length support (163840 tokens)

---

## Component Specifications

### 1. Configuration (`DeepseekV32Config`)

Reference: `deepseek_files/config.json`

```python
class DeepseekV32Config(PretrainedConfig):
    model_type = "deepseek_v32"
```

#### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | 129280 | Vocabulary size |
| `hidden_size` | int | 7168 | Hidden dimension |
| `intermediate_size` | int | 18432 | Dense MLP intermediate size |
| `moe_intermediate_size` | int | 2048 | MoE expert intermediate size |
| `num_hidden_layers` | int | 61 | Number of decoder layers |
| `num_attention_heads` | int | 128 | Number of attention heads |
| `num_key_value_heads` | int | 128 | Number of KV heads (same as attention for MLA) |

#### MLA Parameters

Reference: `model.py:74-79` (ModelArgs)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q_lora_rank` | int | 1536 | LoRA rank for query compression |
| `kv_lora_rank` | int | 512 | LoRA rank for key-value compression |
| `qk_nope_head_dim` | int | 128 | Q/K dimension without position embedding |
| `qk_rope_head_dim` | int | 64 | Q/K dimension with rotary position embedding |
| `v_head_dim` | int | 128 | Value head dimension |

#### MoE Parameters

Reference: `model.py:66-73` (ModelArgs), `model.py:646-709` (Gate class)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_routed_experts` | int | 256 | Total number of routed experts |
| `n_shared_experts` | int | 1 | Number of shared experts (always active) |
| `num_experts_per_tok` | int | 8 | Experts activated per token |
| `n_group` | int | 8 | Number of expert groups |
| `topk_group` | int | 4 | Groups selected in routing |
| `routed_scaling_factor` | float | 2.5 | Scaling factor for routed outputs |
| `scoring_func` | str | "sigmoid" | Scoring function ("softmax" or "sigmoid") |
| `topk_method` | str | "noaux_tc" | Top-k selection method |
| `first_k_dense_replace` | int | 3 | Initial layers using dense MLP |

#### Indexer (DSA) Parameters

Reference: `model.py:87-90` (ModelArgs), `model.py:435-487` (Indexer class)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `index_n_heads` | int | 64 | Indexer attention heads |
| `index_head_dim` | int | 128 | Indexer head dimension |
| `index_topk` | int | 2048 | Top-k tokens for sparse attention |

#### RoPE/YaRN Parameters

Reference: `model.py:80-86` (ModelArgs), `model.py:324-402` (precompute_freqs_cis)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rope_theta` | float | 10000.0 | RoPE base frequency |
| `rope_scaling` | dict | See below | YaRN scaling configuration |

**rope_scaling dict structure:**
```python
{
    "type": "yarn",
    "factor": 40,
    "original_max_position_embeddings": 4096,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "beta_fast": 32,
    "beta_slow": 1
}
```

---

### 2. RMSNorm (`DeepseekV32RMSNorm`)

Reference: `model.py:272-306`

Standard Root Mean Square Layer Normalization.

```python
class DeepseekV32RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states
```

---

### 3. Rotary Embedding (`DeepseekV32RotaryEmbedding`)

Reference: `model.py:324-402` (precompute_freqs_cis), `model.py:405-425` (apply_rotary_emb)

#### YaRN Frequency Computation

The YaRN algorithm adjusts frequencies for extended context:

```python
def precompute_freqs_cis(config):
    dim = config.qk_rope_head_dim  # 64
    base = config.rope_theta  # 10000
    factor = config.rope_scaling["factor"]  # 40
    beta_fast = config.rope_scaling["beta_fast"]  # 32
    beta_slow = config.rope_scaling["beta_slow"]  # 1
    original_seq_len = config.rope_scaling["original_max_position_embeddings"]  # 4096

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))

    if max_seq_len > original_seq_len:
        # Find correction range
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    return freqs
```

#### Two RoPE Formats

**CRITICAL**: The model uses TWO different RoPE formats:

1. **Interleaved** (main attention): Standard format where pairs are adjacent
   - Reference: `model.py:405-425`, `interleaved=True`

2. **Non-interleaved** (indexer): First half real, second half imaginary
   - Reference: `model.py:464, 470`, `interleaved=False`

```python
def apply_rotary_emb(x, freqs_cis, interleaved=True):
    if not interleaved:
        # Rearrange: [r0, r1, ..., i0, i1, ...] -> complex format
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()

    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    y = torch.view_as_real(x * freqs_cis).flatten(-2)

    if not interleaved:
        # Convert back to non-interleaved
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)

    return y
```

---

### 4. Lightning Indexer (`DeepseekV32Indexer`)

Reference: `model.py:435-487`

The Lightning Indexer is the core innovation of DSA. It computes sparse attention indices using the formula:

$$I_{t,s} = \sum_j w^I_{t,j} \cdot \text{ReLU}(q^I_{t,j} \cdot k^I_s)$$

#### Architecture

```
Input: hidden_states [B, S, hidden_size]
       q_compressed [B, S, q_lora_rank]  (from main attention)

Projections:
  - q_b_proj: q_lora_rank -> index_n_heads * index_head_dim
  - k_proj: hidden_size -> index_head_dim (single-head)
  - k_norm: LayerNorm(index_head_dim)
  - weight_proj: hidden_size -> index_n_heads

Output: topk_indices [B, S, index_topk]
```

#### Forward Pass

Reference: `model.py:457-487`

```python
def forward(self, x, qr, start_pos, freqs_cis, mask):
    # 1. Project Q from compressed representation
    q = self.wq_b(qr)  # [B, S, H*D]
    q = q.view(B, S, n_heads, head_dim)

    # 2. Split and apply NON-INTERLEAVED RoPE
    q_pe, q_nope = torch.split(q, [rope_dim, head_dim - rope_dim], dim=-1)
    q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=False)  # KEY: False
    q = torch.cat([q_pe, q_nope], dim=-1)

    # 3. Project K (single-head) with LayerNorm
    k = self.k_norm(self.wk(x))  # [B, S, D]
    k_pe, k_nope = torch.split(k, [rope_dim, head_dim - rope_dim], dim=-1)
    k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, interleaved=False)
    k = torch.cat([k_pe, k_nope], dim=-1)

    # 4. Apply Hadamard transform (optional optimization)
    q = rotate_activation(q)  # Hadamard transform
    k = rotate_activation(k)

    # 5. Compute head weights
    weights = self.weights_proj(x) * (n_heads ** -0.5)

    # 6. Compute index scores: I = Σ w * ReLU(q · k)
    scores = einsum("bshd,btd->bsht", q, k) * softmax_scale
    scores = scores * weights.unsqueeze(-1)
    scores = relu(scores)
    index_scores = scores.sum(dim=2)  # Sum over heads

    # 7. Apply mask and select top-k
    if mask is not None:
        index_scores = index_scores + mask
    topk_indices = index_scores.topk(min(index_topk, T)).indices

    return topk_indices
```

#### Key Implementation Details

1. **Non-interleaved RoPE**: The indexer uses `interleaved=False` for RoPE
2. **Single-head keys**: K is projected to single head, broadcast to multi-head Q
3. **Hadamard transform**: Applied via `fast_hadamard_transform` library
4. **ReLU activation**: Applied BEFORE summing over heads
5. **FP8 quantization**: Optional, requires custom kernels (not in HF impl)

---

### 5. Multi-head Latent Attention (`DeepseekV32Attention`)

Reference: `model.py:498-608` (MLA class)

#### Architecture

```
Q Path (LoRA compressed):
  hidden_states -> q_a_proj -> q_norm -> q_b_proj -> Q
  [B,S,hidden] -> [B,S,q_lora_rank] -> [B,S,H*(qk_nope+qk_rope)]

KV Path (LoRA compressed + rope stream):
  hidden_states -> kv_a_proj -> split -> kv_norm -> kv_b_proj -> K_nope, V
                           \-> k_rope (single-head)
  [B,S,hidden] -> [B,S,kv_lora_rank+rope] -> [B,S,H*(qk_nope+v)]

Output:
  o_proj: H * v_head_dim -> hidden_size
```

#### Projection Dimensions

| Projection | Input | Output |
|------------|-------|--------|
| `q_a_proj` | hidden_size (7168) | q_lora_rank (1536) |
| `q_b_proj` | q_lora_rank (1536) | n_heads * qk_head_dim (128 * 192) |
| `kv_a_proj_with_mqa` | hidden_size (7168) | kv_lora_rank + qk_rope_head_dim (512 + 64) |
| `kv_b_proj` | kv_lora_rank (512) | n_heads * (qk_nope + v) (128 * 256) |
| `o_proj` | n_heads * v_head_dim (128 * 128) | hidden_size (7168) |

#### Softmax Scale with YaRN mscale

Reference: `model.py:533-537`

```python
self.softmax_scale = qk_head_dim ** -0.5  # Base scale
if max_seq_len > original_seq_len:
    mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
    self.softmax_scale = self.softmax_scale * mscale * mscale
```

#### Forward Pass (Prefill vs Decode)

Reference: `model.py:545-608`

**Prefill Path (seq_len > 1):**
```python
# Full MHA computation
q = cat([q_nope, q_pe], dim=-1)
kv = self.wkv_b(kv_compressed)
k_nope, v = split(kv)
k = cat([k_nope, k_pe.expand(n_heads)], dim=-1)

scores = einsum("bshd,bthd->bsht", q, k) * softmax_scale

# Apply indexer mask
topk_indices = self.indexer(x, qr, freqs_cis, mask)
index_mask = full((B, S, T), -inf).scatter(-1, topk_indices, 0)
scores = scores + index_mask.unsqueeze(2)

output = softmax(scores) @ v
```

**Decode Path (seq_len == 1):**
```python
# Efficient MQA using cached compressed KV
wkv_b = self.wkv_b.weight.view(n_heads, qk_nope + v, kv_lora_rank)

# Compute scores via weight sharing
q_nope_proj = einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope])
scores_nope = einsum("bshc,btc->bsht", q_nope_proj, kv_cache)
scores_rope = einsum("bshr,btr->bsht", q_pe, pe_cache)
scores = (scores_nope + scores_rope) * softmax_scale

# Apply indexer and compute output
topk_indices = self.indexer(...)
v = einsum("bshc,hdc->bshd", kv_cache @ wkv_b[:, -v_dim:])
output = softmax(scores) @ v
```

---

### 6. MLP (`DeepseekV32MLP`)

Reference: `model.py:611-643`

Standard SwiGLU MLP:

```python
class DeepseekV32MLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = Linear(hidden_size, intermediate_size)
        self.up_proj = Linear(hidden_size, intermediate_size)
        self.down_proj = Linear(intermediate_size, hidden_size)
        self.act_fn = SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

---

### 7. Gate (`DeepseekV32Gate`)

Reference: `model.py:646-709`

#### Key Features

1. **Sigmoid scoring** (not softmax)
2. **Gate bias** for 7168 hidden_size models
3. **Group-based routing** with `noaux_tc` method

```python
class DeepseekV32Gate(nn.Module):
    def __init__(self, config):
        self.weight = Parameter(empty(n_routed_experts, hidden_size))
        # Bias only for 7168 hidden_size (671B model)
        self.bias = Parameter(zeros(n_routed_experts)) if hidden_size == 7168 else None

    def forward(self, x):
        # 1. Compute scores
        scores = linear(x.float(), self.weight.float())

        # 2. Apply scoring function
        if self.scoring_func == "softmax":
            scores = scores.softmax(dim=-1)
        else:  # sigmoid
            scores = scores.sigmoid()

        original_scores = scores

        # 3. Apply bias
        if self.bias is not None:
            scores = scores + self.bias

        # 4. Group-based selection (noaux_tc)
        if self.n_groups > 1:
            scores = scores.view(B, n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                # Use top-2 sum when bias present
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)

            group_idx = group_scores.topk(topk_groups)[1]
            mask = ones(B, n_groups, dtype=bool).scatter(1, group_idx, False)
            scores = scores.masked_fill(mask.unsqueeze(-1), -inf).flatten(1)

        # 5. Select top-k experts
        indices = scores.topk(top_k)[1]
        weights = original_scores.gather(1, indices)

        # 6. Normalize for sigmoid
        if self.scoring_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True)

        weights = weights * routed_scaling_factor
        return weights, indices
```

---

### 8. MoE (`DeepseekV32MoE`)

Reference: `model.py:747-804`

```python
class DeepseekV32MoE(nn.Module):
    def __init__(self, config):
        self.gate = DeepseekV32Gate(config)
        self.experts = ModuleList([Expert(config) for _ in range(n_routed_experts)])
        self.shared_experts = MLP(hidden_size, n_shared_experts * moe_inter_dim)

    def forward(self, x):
        shape = x.size()
        x = x.view(-1, hidden_size)

        weights, indices = self.gate(x)

        # Route tokens to experts
        y = zeros_like(x, dtype=float32)
        counts = bincount(indices.flatten(), minlength=n_routed_experts)

        for i, expert in enumerate(self.experts):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]

        # Add shared experts
        y = y + self.shared_experts(x)

        return y.view(shape)
```

---

### 9. Decoder Layer (`DeepseekV32DecoderLayer`)

Reference: `model.py:807-851` (Block class)

```python
class DeepseekV32DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        self.self_attn = DeepseekV32Attention(config, layer_idx)

        # Dense MLP for first k layers, MoE for rest
        if layer_idx < config.first_k_dense_replace:
            self.mlp = DeepseekV32MLP(config)
        else:
            self.mlp = DeepseekV32MoE(config)

        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(self, x, attention_mask, position_embeddings, ...):
        # Pre-norm attention
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, position_embeddings, attention_mask, ...)
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x
```

---

### 10. Full Model (`DeepseekV32Model`, `DeepseekV32ForCausalLM`)

Reference: `model.py:854-913` (Transformer class)

```python
class DeepseekV32Model(PreTrainedModel):
    def __init__(self, config):
        self.embed_tokens = Embedding(vocab_size, hidden_size)
        self.layers = ModuleList([
            DeepseekV32DecoderLayer(config, i)
            for i in range(num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.rotary_emb = DeepseekV32RotaryEmbedding(config)

    def forward(self, input_ids, attention_mask, position_ids, ...):
        x = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(x, position_ids)

        for layer in self.layers:
            x = layer(x, attention_mask, position_embeddings, ...)

        x = self.norm(x)
        return x


class DeepseekV32ForCausalLM(PreTrainedModel):
    def __init__(self, config):
        self.model = DeepseekV32Model(config)
        self.lm_head = Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, labels=None, ...):
        hidden_states = self.model(input_ids, ...)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = CrossEntropyLoss()(logits[..., :-1, :], labels[..., 1:])

        return CausalLMOutputWithPast(loss=loss, logits=logits, ...)
```

---

## Key Differences from DeepSeek V2

| Aspect | DeepSeek V2 | DeepSeek V3.2 |
|--------|-------------|---------------|
| Attention | Dense MLA | Sparse MLA with DSA |
| Indexer | None | Lightning Indexer |
| MoE scoring | softmax | sigmoid |
| MoE routing | greedy/group_limited | noaux_tc |
| Gate bias | None | Yes (for 7168 hidden) |
| RoPE in indexer | N/A | Non-interleaved |
| Context length | Shorter | 163840 with YaRN |

---

## Optional Features

### Hadamard Transform

Reference: `model.py:428-432`, `model.py:472-473`

Requires `fast_hadamard_transform` package:

```python
from fast_hadamard_transform import hadamard_transform

def rotate_activation(x):
    hidden_size = x.size(-1)
    return hadamard_transform(x.bfloat16(), scale=hidden_size ** -0.5)
```

### FP8 Quantization

Reference: `model.py:474-480`

Requires custom `tilelang` kernels (not implemented in HuggingFace):

```python
q_fp8, q_scale = act_quant(q, block_size, scale_fmt)
k_fp8, k_scale = act_quant(k, block_size, scale_fmt)
index_score = fp8_index(q_fp8, weights, k_cache, k_scale_cache)
```

---

## Testing Strategy

1. **Config Test**: Load official `config.json`, verify all params parse correctly
2. **Import Test**: Verify clean imports with no errors
3. **Small Model Test**: Instantiate with reduced config, run forward pass
4. **Shape Tests**: Verify all tensor shapes match expected dimensions
5. **Reference Comparison**: Compare outputs with reference implementation (if weights available)

---

## File Structure

```
src/transformers/models/deepseek_v32/
├── __init__.py                    # Module exports
├── configuration_deepseek_v32.py  # Generated from modular
├── modeling_deepseek_v32.py       # Generated from modular
├── modular_deepseek_v32.py        # Primary implementation file
├── SPEC.md                        # This specification
└── deepseek_files/                # Reference files
    ├── config.json                # Official config
    └── inference/
        └── model.py               # Reference implementation
```

---

## Implementation Checklist

- [ ] `DeepseekV32Config` - All parameters from config.json
- [ ] `DeepseekV32RMSNorm` - Standard RMSNorm
- [ ] `DeepseekV32RotaryEmbedding` - YaRN support, both RoPE formats
- [ ] `DeepseekV32MLP` - SwiGLU activation
- [ ] `DeepseekV32Expert` - Single expert MLP
- [ ] `DeepseekV32Gate` - Sigmoid scoring, noaux_tc, gate bias
- [ ] `DeepseekV32MoE` - Expert routing with shared experts
- [ ] `DeepseekV32Indexer` - Lightning Indexer with non-interleaved RoPE
- [ ] `DeepseekV32Attention` - MLA with DSA, YaRN mscale
- [ ] `DeepseekV32DecoderLayer` - Dense/MoE MLP selection
- [ ] `DeepseekV32Model` - Base model
- [ ] `DeepseekV32ForCausalLM` - Causal LM head
- [ ] `DeepseekV32ForSequenceClassification` - Classification head
