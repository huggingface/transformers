# DeepSeek V3.2 API Reference

This document describes the **capabilities and API** exposed by the DeepSeek V3.2 HuggingFace implementation. Training recipes and configs are managed separately in research-infra.

## Installation

```bash
pip install git+https://github.com/lyfegame/transformers.git@shuyingl/deepseek-v3.2-test
pip install fast-hadamard-transform  # Optional but recommended for performance
```

---

## Model Classes

### DeepseekV32ForCausalLM

Main class for causal language modeling.

```python
from transformers import DeepseekV32ForCausalLM, DeepseekV32Config

model = DeepseekV32ForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V3.2-Exp")
# or
model = DeepseekV32ForCausalLM(config)
```

### Output Classes

```python
from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32ModelOutput,
    DeepseekV32CausalLMOutput,
)
```

---

## Configuration Options

### Sparse Attention Control

| Config Parameter | Type | Default | Description |
|-----------------|------|---------|-------------|
| `use_sparse_attention` | bool | True | Enable/disable Lightning Indexer sparse attention |
| `index_n_heads` | int | 64 | Number of indexer attention heads |
| `index_head_dim` | int | 128 | Dimension per indexer head |
| `index_topk` | int | 2048 | Number of tokens selected per query |

```python
# Enable sparse attention (default)
config.use_sparse_attention = True

# Disable sparse attention (dense mode)
config.use_sparse_attention = False
```

---

## Forward Pass API

### Standard Forward

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,  # Optional, for computing loss
)

# Returns DeepseekV32CausalLMOutput:
outputs.loss        # Language modeling loss (if labels provided)
outputs.logits      # [batch, seq, vocab_size]
```

### Indexer Training Outputs

Two additional forward pass parameters expose indexer internals for training:

```python
outputs = model(
    input_ids=input_ids,
    output_indexer_scores=True,      # Return raw indexer scores
    output_indexer_kl_target=True,   # Return KL divergence target
)

# Additional outputs:
outputs.indexer_scores     # Tuple of [batch, seq, seq] per layer
outputs.indexer_kl_targets # Tuple of [batch, seq, seq] per layer
```

| Parameter | Type | Output Shape | Description |
|-----------|------|--------------|-------------|
| `output_indexer_scores` | bool | `[batch, seq, seq]` per layer | Raw indexer `I_{t,s}` scores before top-k selection |
| `output_indexer_kl_target` | bool | `[batch, seq, seq]` per layer | KL target `p_{t,:}` (L1-normalized attention, detached) |

**Memory Note**: `output_indexer_kl_target` computes the target efficiently inside the forward pass. It sums attention across heads and L1-normalizes, returning only `[batch, seq, seq]` instead of the full `[batch, heads, seq, seq]`.

---

## Helper Functions

### compute_indexer_kl_loss

Computes KL-divergence loss between indexer predictions and attention distribution.

```python
from transformers.models.deepseek_v32.modeling_deepseek_v32 import compute_indexer_kl_loss

outputs = model(
    input_ids,
    output_indexer_scores=True,
    output_indexer_kl_target=True,
)

kl_loss = compute_indexer_kl_loss(
    outputs.indexer_scores,      # Indexer predictions
    outputs.indexer_kl_targets,  # Target distribution
)
# Returns: scalar tensor, averaged across layers
```

**Formula** (from tech report):
```
L_I = sum_t D_KL(p_{t,:} || Softmax(I_{t,:}))
```

---

## Parameter Freezing Utilities

### Freeze Indexer

```python
for name, param in model.named_parameters():
    if "indexer" in name:
        param.requires_grad = False
```

### Freeze Everything Except Indexer

```python
for name, param in model.named_parameters():
    if "indexer" not in name:
        param.requires_grad = False
```

### Get Parameter Counts

```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
indexer = sum(p.numel() for n, p in model.named_parameters() if "indexer" in n)

print(f"Total: {total:,}")
print(f"Trainable: {trainable:,}")
print(f"Indexer: {indexer:,}")
```

---

## Indexer Parameter Names

For PEFT/LoRA targeting (names match official DeepSeek weights):

```python
# Indexer projection layers
"self_attn.indexer.wq_b"        # Query projection (from compressed rep)
"self_attn.indexer.wk"          # Key projection
"self_attn.indexer.weights_proj" # Per-head weight projection
"self_attn.indexer.k_norm"      # Key LayerNorm
```

---

## Usage Examples

### Example 1: Standard Inference

```python
from transformers import DeepseekV32ForCausalLM, AutoTokenizer

model = DeepseekV32ForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V3.2-Exp")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3.2-Exp")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### Example 2: Get Indexer Scores for Analysis

```python
outputs = model(
    input_ids,
    output_indexer_scores=True,
)

# Analyze which tokens the indexer selects
for layer_idx, scores in enumerate(outputs.indexer_scores):
    topk_indices = scores.topk(k=64, dim=-1).indices
    print(f"Layer {layer_idx}: top tokens = {topk_indices[0, 0, :10]}")
```

### Example 3: Compute KL Loss (for indexer training)

```python
from transformers.models.deepseek_v32.modeling_deepseek_v32 import compute_indexer_kl_loss

# Freeze non-indexer params
for name, param in model.named_parameters():
    if "indexer" not in name:
        param.requires_grad = False

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3
)

# Training step
outputs = model(
    input_ids,
    output_indexer_scores=True,
    output_indexer_kl_target=True,
)
kl_loss = compute_indexer_kl_loss(outputs.indexer_scores, outputs.indexer_kl_targets)

kl_loss.backward()
optimizer.step()
optimizer.zero_grad()
```

### Example 4: Dense Mode (disable sparse attention)

```python
from transformers import DeepseekV32Config, DeepseekV32ForCausalLM

config = DeepseekV32Config.from_pretrained("deepseek-ai/DeepSeek-V3.2-Exp")
config.use_sparse_attention = False  # Use dense attention

model = DeepseekV32ForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V3.2-Exp",
    config=config,
)
```

### Example 5: Dual LoRA Training (LLM LoRA + Indexer LoRA)

Two separate gradient paths for training main model and indexer independently:

```python
from peft import LoraConfig, get_peft_model
import torch

# Define LoRA targets
llm_lora_targets = [
    "q_a_proj", "q_b_proj",
    "kv_a_proj_with_mqa", "kv_b_proj",
    "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
indexer_lora_targets = [
    "indexer.q_b_proj",
    "indexer.k_proj",
    "indexer.weight_proj",
]

# Apply LoRA to both
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=llm_lora_targets + indexer_lora_targets,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model_with_lora = get_peft_model(model, lora_config)

# Create separate optimizers for dual gradient paths
llm_lora_params = [p for n, p in model_with_lora.named_parameters()
                   if p.requires_grad and "indexer" not in n]
indexer_lora_params = [p for n, p in model_with_lora.named_parameters()
                       if p.requires_grad and "indexer" in n]

llm_optimizer = torch.optim.AdamW(llm_lora_params, lr=1e-4)
indexer_optimizer = torch.optim.AdamW(indexer_lora_params, lr=1e-3)

# Training step with dual gradient paths
outputs = model_with_lora(input_ids, labels=labels)
lm_loss = outputs.loss

# Placeholder for actual KL loss computation
# kl_loss = compute_indexer_kl_loss(outputs.indexer_scores, outputs.indexer_kl_targets)
kl_loss = outputs.logits.var() * 0.001  # Placeholder

# Path 1: LM loss -> LLM LoRA
llm_optimizer.zero_grad()
lm_loss.backward(retain_graph=True)
llm_optimizer.step()

# Path 2: KL loss -> Indexer LoRA
indexer_optimizer.zero_grad()
kl_loss.backward()
indexer_optimizer.step()
```

### Example 6: Modal GPU Testing

Run comprehensive verification on Modal:

```bash
# Install Modal
pip install modal
modal setup

# Run with random weights (small config)
modal run --detach scripts/modal_verify_deepseek_v32.py --config small

# Run with full model checkpoint
modal run --detach scripts/modal_verify_deepseek_v32.py --checkpoint deepseek-ai--DeepSeek-V3.2_bf16

# Check logs
modal app logs <APP_ID>
```

Modal image configuration with fast-hadamard-transform:

```python
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "clang")
    .pip_install("torch>=2.4.0", "accelerate", "peft", ...)
    # GPU required for CUDA compilation of fast-hadamard-transform
    .run_commands("pip install fast-hadamard-transform", gpu="H100")
    .run_commands("pip install git+https://github.com/lyfegame/transformers.git@shuyingl/deepseek-v3.2-test")
)
```

---

## Technical Notes

### KL Loss Formula (from Tech Report)

The indexer KL loss matches **Equation 3** from the DeepSeek V3.2 technical report (arXiv 2512.02556):

```
ℒ_I = Σ_t D_KL(p_{t,:} ∥ Softmax(I_{t,:}))
```

Where:
- `I_{t,s}` = raw indexer output scores (returned by `output_indexer_scores=True`)
- `p_{t,:}` = target distribution (returned by `output_indexer_kl_target=True`)

The target distribution `p_{t,:}` is computed per the tech report:
> "for the t-th query token, we first aggregate the main attention scores by **summing across all attention heads**. This sum is then **L1-normalized** along the sequence dimension to produce a target distribution p_{t,:} ∈ ℝ^t"

### Two Exposed Losses for LoRA Integration

The implementation exposes **two separate losses** for flexible training:

| Loss | Source | Use Case |
|------|--------|----------|
| **LM Loss** | `outputs.loss` (when `labels` provided) | Standard language modeling |
| **Indexer KL Loss** | `compute_indexer_kl_loss(outputs.indexer_scores, outputs.indexer_kl_targets)` | Indexer alignment |

**Training scenarios:**

1. **Main model LoRA only** (frozen indexer):
   ```python
   loss = outputs.loss  # LM loss only
   ```

2. **Indexer training only** (per tech report warm-up):
   ```python
   kl_loss = compute_indexer_kl_loss(outputs.indexer_scores, outputs.indexer_kl_targets)
   ```

3. **Joint training** (main model + indexer LoRA):
   ```python
   total_loss = outputs.loss + alpha * compute_indexer_kl_loss(...)
   ```

### Non-Interleaved RoPE

The indexer uses **non-interleaved** RoPE (different from MLA which uses interleaved). This is handled internally - no user action required.

### Causal Masking

Both `indexer_scores` and `indexer_kl_targets` are causally masked - position `t` can only attend to positions `s <= t`.

### Gradient Flow

- `indexer_scores`: Gradients flow back to indexer parameters
- `indexer_kl_targets`: **Detached** - serves as target, no gradient flow
- When `output_indexer_kl_target=True`, full attention is computed for the target but not stored (memory efficient)

### Memory Efficiency

| Output | Memory per Layer |
|--------|-----------------|
| `indexer_scores` | `[B, S, S]` = `B × S² × 4 bytes` |
| `indexer_kl_targets` | `[B, S, S]` = `B × S² × 4 bytes` |
| Full attention (NOT returned) | `[B, H, S, S]` = `B × H × S² × 4 bytes` |

For 128K context with 128 heads, returning full attention would require ~8TB per layer. The API returns only the aggregated target.

---

## Files Reference

| File | Description |
|------|-------------|
| `modeling_deepseek_v32.py` | Model implementation with indexer training API |
| `configuration_deepseek_v32.py` | Configuration class |
| `compute_indexer_kl_loss()` | Helper function for KL loss |
| `DeepseekV32ModelOutput` | Extended output dataclass |
| `DeepseekV32CausalLMOutput` | Extended CausalLM output dataclass |

---

## Compatibility

- **PyTorch**: >= 2.4.0
- **CUDA**: Tested on H100/H200
- **PEFT**: Compatible (target indexer modules for LoRA)
- **DeepSpeed ZeRO**: Compatible

### Attention Backend Support

DeepSeek V3.2 supports different attention backends depending on the mode:

| Mode | Condition | Attention Backend | Notes |
|------|-----------|-------------------|-------|
| **Sparse Prefill** | `use_sparse_attention=True` AND `seq_len > 1` | Eager (PyTorch) | Matches official DeepSeek code |
| **Decode** | `seq_len == 1` (autoregressive) | Flash/SDPA/Eager | Delegates to V3 parent |
| **Dense Mode** | `use_sparse_attention=False` | Flash/SDPA/Eager | Delegates to V3 parent |

**Why sparse attention uses eager computation:**
- The official DeepSeek V3.2 code uses eager attention with dynamic sparse masks
- Flash Attention requires **block-sparse** patterns known at compile time
- The Lightning Indexer produces **dynamic** sparse patterns that vary per input
- Custom block-sparse kernels would be needed for flash attention + DSA

**Practical implications:**
```python
# Sparse attention (prefill) - eager attention
config.use_sparse_attention = True
outputs = model(input_ids)  # Uses eager attention during prefill

# Dense mode - flash attention supported
config.use_sparse_attention = False
model = DeepseekV32ForCausalLM.from_pretrained(
    "...",
    config=config,
    attn_implementation="flash_attention_2"  # Works in dense mode
)

# Generation (decode phase) - flash attention supported
# Even with use_sparse_attention=True, decode uses V3's attention
# which supports flash attention
outputs = model.generate(input_ids, max_new_tokens=100)
```

### Flash Attention vs DeepSeek Sparse Attention (DSA)

**Important**: Flash Attention and DSA are **different mechanisms**:

| Mechanism | What it does | Complexity |
|-----------|--------------|------------|
| **Flash Attention** | Optimized CUDA kernel for dense attention (memory efficient via tiling) | O(N²) compute, O(N) memory |
| **DeepSeek Sparse Attention (DSA)** | Sparse attention via Lightning Indexer, selects top-k tokens | O(N×k) compute |

**Summary:**
- **Prefill with sparse attention**: Eager attention (matches official code)
- **Decode**: Flash attention supported (delegates to V3)
- **Dense mode**: Flash attention supported (delegates to V3)
