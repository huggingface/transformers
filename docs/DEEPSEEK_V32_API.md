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
    CausalLMOutputWithIndexer,       # For DeepseekV32ForCausalLM
)
from transformers.modeling_outputs import BaseModelOutputWithPast  # For DeepseekV32Model
```

**Note:** `DeepseekV32Model` returns standard `BaseModelOutputWithPast`. Indexer outputs (scores, KL targets) are stored internally on the model instance for use by `DeepseekV32ForCausalLM`.

---

## Configuration Options

### Official Model Configuration

The official DeepSeek V3.2 model uses this configuration (from HuggingFace):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hidden_size` | 7168 | Model hidden dimension |
| `num_hidden_layers` | 61 | Number of transformer layers |
| `num_attention_heads` | 128 | Number of attention heads |
| `num_key_value_heads` | 128 | Same as attention heads (no GQA) |
| `index_n_heads` | 64 | Indexer attention heads |
| `index_head_dim` | 128 | Indexer head dimension |
| `index_topk` | 2048 | Tokens selected per query |
| `q_lora_rank` | 1536 | Query LoRA rank |
| `kv_lora_rank` | 512 | KV LoRA rank |
| `qk_rope_head_dim` | 64 | RoPE head dimension |
| `qk_nope_head_dim` | 128 | Non-RoPE head dimension |
| `v_head_dim` | 128 | Value head dimension |
| `n_routed_experts` | 256 | Number of routed MoE experts |
| `n_shared_experts` | 1 | Number of shared experts |
| `num_experts_per_tok` | 8 | Experts activated per token |
| `first_k_dense_replace` | 3 | First 3 layers use dense MLP |
| `max_position_embeddings` | 163840 | Maximum context length |
| `vocab_size` | 129280 | Vocabulary size |

### Sparse Attention Control

| Config Parameter | Type | Default | Description |
|-----------------|------|---------|-------------|
| `use_sparse_attention` | bool | True | Enable/disable Lightning Indexer sparse attention |
| `index_n_heads` | int | 64 | Number of indexer attention heads |
| `index_head_dim` | int | 128 | Dimension per indexer head |
| `index_topk` | int | 2048 | Number of tokens selected per query |
| `indexer_kl_coef` | float | 0.0 | Coefficient for automatic KL loss in combined loss |

```python
# Enable sparse attention (default)
config.use_sparse_attention = True

# Disable sparse attention (dense mode)
config.use_sparse_attention = False

# Enable automatic KL loss computation (added to combined loss)
config.indexer_kl_coef = 0.1  # outputs.loss = lm_loss + 0.1 * indexer_kl_loss
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

# Returns CausalLMOutputWithIndexer:
outputs.loss             # Combined loss: lm_loss + indexer_kl_coef * indexer_kl_loss
outputs.lm_loss          # Pure language modeling loss (if labels provided)
outputs.indexer_kl_loss  # KL divergence loss (if indexer outputs requested)
outputs.logits           # [batch, seq, vocab_size]
```

### Indexer Training Outputs

The `CausalLMOutputWithIndexer` extends `CausalLMOutputWithPast` with two additional fields for training:

```python
outputs = model(input_ids, labels=labels)

# Standard outputs (inherited from CausalLMOutputWithPast):
outputs.loss             # Combined loss: lm_loss + indexer_kl_coef * indexer_kl_loss
outputs.logits           # [batch, seq, vocab_size]
outputs.past_key_values  # KV cache for generation
outputs.hidden_states    # Optional tuple of hidden states
outputs.attentions       # Optional tuple of attention weights

# V3.2 specific outputs for training:
outputs.lm_loss          # Pure language modeling loss
outputs.indexer_kl_loss  # KL divergence loss (when indexer_kl_coef > 0)
```

| Field | Type | Description |
|-------|------|-------------|
| `lm_loss` | `torch.FloatTensor` | Pure cross-entropy loss for language modeling |
| `indexer_kl_loss` | `torch.FloatTensor` | KL divergence loss `D_KL(attention_dist \|\| indexer_dist)` |

**KL Loss Computation**: When `config.indexer_kl_coef > 0`, indexer scores and attention targets are computed internally during the forward pass and used to compute `indexer_kl_loss`. The combined loss is: `loss = lm_loss + indexer_kl_coef * indexer_kl_loss`.

**Memory Note**: The KL target is computed efficiently by summing attention across heads and L1-normalizing, never storing the full `[batch, heads, seq, seq]` attention tensor.

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

These names match the official DeepSeek V3.2 checkpoint (verified against `model.safetensors.index.json`):

```python
# Indexer projection layers (use these for LoRA target_modules)
"indexer.wq_b"         # Query projection (from compressed rep)
"indexer.wk"           # Key projection
"indexer.weights_proj" # Per-head weight projection
"indexer.k_norm"       # Key LayerNorm (weight + bias, not typically for LoRA)
```

**Full parameter paths** (matches official checkpoint):
```
model.layers.{i}.self_attn.indexer.wq_b.weight
model.layers.{i}.self_attn.indexer.wk.weight
model.layers.{i}.self_attn.indexer.weights_proj.weight
model.layers.{i}.self_attn.indexer.k_norm.weight
model.layers.{i}.self_attn.indexer.k_norm.bias
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

### Example 2: Analyze Indexer Behavior

```python
# Enable KL loss to trigger indexer score computation
model.config.indexer_kl_coef = 0.1

outputs = model(input_ids, labels=labels)

# Indexer scores are computed internally and used for KL loss
# The KL loss indicates how well the indexer matches the full attention
print(f"Indexer KL loss: {outputs.indexer_kl_loss.item():.4f}")
print(f"LM loss: {outputs.lm_loss.item():.4f}")

# For detailed analysis, use hooks on the indexer layers
# (see DeepseekV32Indexer for the internal score computation)
```

### Example 3: Indexer Training with KL Loss

```python
# Freeze non-indexer params
for name, param in model.named_parameters():
    if "indexer" not in name:
        param.requires_grad = False

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3
)

# Enable KL loss computation
model.config.indexer_kl_coef = 1.0

# Training step - KL loss is computed automatically
outputs = model(input_ids, labels=labels)
kl_loss = outputs.indexer_kl_loss  # Use the automatically computed KL loss

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
    "indexer.wq_b",         # Query projection
    "indexer.wk",           # Key projection
    "indexer.weights_proj", # Per-head weight projection
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

# Enable KL loss computation
model_with_lora.base_model.model.config.indexer_kl_coef = 1.0

# Create separate optimizers for dual gradient paths
llm_lora_params = [p for n, p in model_with_lora.named_parameters()
                   if p.requires_grad and "indexer" not in n]
indexer_lora_params = [p for n, p in model_with_lora.named_parameters()
                       if p.requires_grad and "indexer" in n]

llm_optimizer = torch.optim.AdamW(llm_lora_params, lr=1e-4)
indexer_optimizer = torch.optim.AdamW(indexer_lora_params, lr=1e-3)

# Training step with dual gradient paths
# indexer_kl_coef > 0 enables automatic KL loss computation
outputs = model_with_lora(input_ids, labels=labels)

# Get separate losses from output
lm_loss = outputs.lm_loss
indexer_kl_loss = outputs.indexer_kl_loss  # Real KL loss, not placeholder!

# Path 1: LM loss -> LLM LoRA
llm_optimizer.zero_grad()
lm_loss.backward(retain_graph=True)
llm_optimizer.step()

# Path 2: KL loss -> Indexer LoRA
indexer_optimizer.zero_grad()
indexer_kl_loss.backward()
indexer_optimizer.step()
```

---

## Technical Notes

### KL Loss Formula (from Tech Report)

The indexer KL loss matches **Equation 3** from the DeepSeek V3.2 technical report (arXiv 2512.02556):

```
ℒ_I = Σ_t D_KL(p_{t,:} ∥ Softmax(I_{t,:}))
```

Where:
- `I_{t,s}` = raw indexer output scores (computed internally)
- `p_{t,:}` = target distribution (computed internally from attention weights)

The target distribution `p_{t,:}` is computed per the tech report:
> "for the t-th query token, we first aggregate the main attention scores by **summing across all attention heads**. This sum is then **L1-normalized** along the sequence dimension to produce a target distribution p_{t,:} ∈ ℝ^t"

When `config.indexer_kl_coef > 0`, both `I_{t,s}` and `p_{t,:}` are computed automatically during the forward pass and used to compute `outputs.indexer_kl_loss`.

### Two Exposed Losses for LoRA Integration

The implementation exposes **separate losses** for flexible training:

| Loss | Source | Use Case |
|------|--------|----------|
| **Combined Loss** | `outputs.loss` | `lm_loss + indexer_kl_coef * indexer_kl_loss` |
| **LM Loss** | `outputs.lm_loss` | Pure language modeling loss |
| **Indexer KL Loss** | `outputs.indexer_kl_loss` | KL divergence loss for indexer |

**Training scenarios:**

1. **Main model LoRA only** (frozen indexer, `indexer_kl_coef=0`):
   ```python
   model.config.indexer_kl_coef = 0.0
   outputs = model(input_ids, labels=labels)
   loss = outputs.loss  # Same as outputs.lm_loss when indexer_kl_coef=0
   ```

2. **Indexer training only** (per tech report warm-up):
   ```python
   model.config.indexer_kl_coef = 1.0
   outputs = model(input_ids, labels=labels)
   kl_loss = outputs.indexer_kl_loss
   ```

3. **Joint training** (main model + indexer LoRA):
   ```python
   # Option A: Use combined loss with config
   model.config.indexer_kl_coef = 0.1
   outputs = model(input_ids, labels=labels)
   loss = outputs.loss  # lm_loss + 0.1 * indexer_kl_loss

   # Option B: Manual combination
   loss = outputs.lm_loss + alpha * outputs.indexer_kl_loss
   ```

4. **Dual LoRA with separate backward passes**:
   ```python
   model.config.indexer_kl_coef = 1.0  # Enable KL loss
   outputs = model(input_ids, labels=labels)
   outputs.lm_loss.backward(retain_graph=True)  # -> LLM LoRA
   outputs.indexer_kl_loss.backward()            # -> Indexer LoRA
   ```

### Non-Interleaved RoPE

The indexer uses **non-interleaved** RoPE (different from MLA which uses interleaved). This is handled internally - no user action required.

### Causal Masking

Both indexer scores and KL targets (computed internally) are causally masked - position `t` can only attend to positions `s <= t`.

### Gradient Flow (Independent Paths)

The implementation provides **two independent gradient paths** for flexible training:

| Loss | Gradient Path | Parameters Updated |
|------|---------------|-------------------|
| `lm_loss` | `logits` → `lm_head` → `hidden_states` → all layers | All model parameters (attention, MLP, embeddings, lm_head) |
| `indexer_kl_loss` | `indexer_scores` → `LightningIndexer` | **Only** indexer parameters (`indexer.wq_b`, `indexer.wk`, `indexer.weights_proj`) |

**Why are they independent?**
- The KL target (`indexer_kl_target`) is **detached** (line 724 in modular_deepseek_v32.py) - it receives no gradients
- `indexer_kl_loss = KL(softmax(indexer_scores) || detached_attention_target)`
- Gradients from `indexer_kl_loss` only flow through `indexer_scores` → indexer parameters

**Dual backward example:**
```python
outputs = model(input_ids, labels=labels)

# These backward passes update DIFFERENT parameters:
outputs.lm_loss.backward(retain_graph=True)  # Updates: attention, MLP, lm_head, embeddings
outputs.indexer_kl_loss.backward()            # Updates: ONLY indexer params
```

This enables dual-LoRA training where LM and indexer have separate optimizers/learning rates.

### Memory Efficiency

The implementation avoids storing large intermediate tensors:

| Internal Computation | Memory Usage |
|---------------------|--------------|
| Indexer scores (per layer) | `[B, S, S]` = `B × S² × 4 bytes` |
| KL target (per layer) | `[B, S, S]` = `B × S² × 4 bytes` |
| Full attention (NOT stored) | `[B, H, S, S]` = `B × H × S² × 4 bytes` |

For 128K context with 128 heads, storing full attention would require ~8TB per layer. The implementation aggregates attention across heads immediately to compute the KL target efficiently.

---

## Files Reference

| Location | File | Description |
|----------|------|-------------|
| `src/transformers/models/deepseek_v32/` | `modular_deepseek_v32.py` | Source of truth for model implementation |
| `src/transformers/models/deepseek_v32/` | `modeling_deepseek_v32.py` | Auto-generated model file |
| `src/transformers/models/deepseek_v32/` | `configuration_deepseek_v32.py` | Configuration class |

**Key exports from `modeling_deepseek_v32.py`:**
- `DeepseekV32ForCausalLM` - Main model class
- `DeepseekV32Model` - Base transformer model (returns `BaseModelOutputWithPast`)
- `DeepseekV32Config` - Configuration class
- `CausalLMOutputWithIndexer` - Output dataclass for DeepseekV32ForCausalLM (extends `CausalLMOutputWithPast`)

---

## Distributed Training

### FSDP / DeepSpeed ZeRO-3

DeepSeek V3.2 uses the **standard HuggingFace MoE pattern** with 3D tensor expert weights (same as Mixtral, Qwen2-MoE, DeepSeek V3). This is fully compatible with FSDP and DeepSpeed ZeRO-3.

#### How It Works

Expert parameters are stored as 3D tensors with shape `[num_experts, output_dim, input_dim]`:
- FSDP shards the **entire 3D tensor** across GPUs (not individual experts)
- ZeRO-3 partitions the 3D tensors automatically
- Indexing `tensor[expert_idx]` retrieves a local slice without triggering collective operations
- This avoids synchronization issues that can occur with `nn.ModuleList`-based implementations

#### Usage with FSDP

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import DeepseekV32ForCausalLM

model = DeepseekV32ForCausalLM(config)

# Wrap the model - FSDP shards the 3D expert tensors automatically
model = FSDP(model)
```

#### Usage with DeepSpeed ZeRO-3

```python
import deepspeed
from transformers import DeepseekV32ForCausalLM

ds_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_param": {"device": "cpu"},  # Optional CPU offload
    },
    "bf16": {"enabled": True},
    "train_batch_size": 8,
    "gradient_accumulation_steps": 4,
}

model = DeepseekV32ForCausalLM(config)
model, optimizer, _, _ = deepspeed.initialize(model=model, config=ds_config)
# ZeRO-3 partitions expert tensors automatically
```

#### Usage with HuggingFace Trainer

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",
    fsdp="full_shard",  # Enable FSDP
    fsdp_config={
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
    },
    bf16=True,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

Or with DeepSpeed:

```python
training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="ds_config.json",  # Your DeepSpeed config with stage 3
    bf16=True,
    gradient_checkpointing=True,
)
```

#### Why 3D Tensors (Not nn.ModuleList)

The MoE implementation uses 3D parameter tensors instead of `nn.ModuleList[nn.Linear]`:

| Aspect | 3D Tensors (Used) | nn.ModuleList (Not Used) |
|--------|-------------------|--------------------------|
| **FSDP behavior** | Shards entire tensor | Wraps each expert module |
| **Indexing** | Local slice, no collective | Each `.forward()` triggers AllGather |
| **Conditional routing** | Safe - all ranks process same tensor | **Dangerous** - ranks may skip different experts |
| **Collective sync** | Implicit via tensor sharding | Explicit per-expert AllGather |

With `nn.ModuleList`, if different ranks skip different experts due to routing, you get NCCL deadlocks (ranks waiting forever for AllGather). The 3D tensor approach avoids this entirely.

#### Key Design Decisions

| Aspect | Implementation | Benefit |
|--------|---------------|---------|
| Expert storage | 3D `nn.Parameter` tensors | FSDP shards entire tensor, no per-expert AllGather |
| Expert computation | `F.linear(input, tensor[idx])` | Local slice from sharded tensor |
| Device management | No explicit `.to()` calls | Frameworks handle placement |
| Gradient sync | Automatic via framework | No manual `all_reduce` needed |

### Meta Device Support (Memory-Efficient Initialization)

DeepSeek V3.2 supports **meta device initialization** for memory-efficient loading of large models with FSDP and DeepSpeed ZeRO-3. This allows creating a 671B parameter model without allocating any memory until the distributed framework shards it.

#### The Problem

Creating a 671B model on CPU before FSDP/ZeRO-3 can shard it requires ~1.3TB of RAM (671B × 2 bytes for bf16), which exceeds typical CPU memory.

#### The Solution

```python
from accelerate import init_empty_weights
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import DeepseekV32ForCausalLM, DeepseekV32Config

config = DeepseekV32Config.from_pretrained("deepseek-ai/DeepSeek-V3.2-Exp")

# Step 1: Create model on meta device (0 bytes memory)
with init_empty_weights():
    model = DeepseekV32ForCausalLM(config)

# All parameters are now on meta device - just tensor metadata, no memory allocated
for name, param in model.named_parameters():
    assert param.device.type == "meta"

# Step 2: Wrap with FSDP - parameters are materialized shard-by-shard
model = FSDP(
    model,
    param_init_fn=lambda m: m.to_empty(device=torch.cuda.current_device(), recurse=False),
    sync_module_states=True,  # Rank 0 broadcasts to other ranks
    # ... other FSDP config
)

# Step 3: Load checkpoint directly to FSDP shards
# (weights are loaded shard-by-shard, never materializing full model on one device)
```

#### How It Works

The implementation uses `torch.empty()` instead of `torch.ones()` / `torch.zeros()` for all parameter and buffer initialization:

| Component | Pattern | Meta-Compatible |
|-----------|---------|----------------|
| `DeepseekV32RMSNorm.weight` | `torch.empty(hidden_size)` | ✅ |
| `DeepseekV32TopkRouter.weight` | `torch.empty(n_experts, hidden_size)` | ✅ |
| `DeepseekV32TopkRouter.e_score_correction_bias` | `torch.zeros(n_experts)` | ✅ (inherited from V3) |
| `DeepseekV32NaiveMoe.gate_up_proj` | `torch.empty(n_experts, 2*intermediate, hidden)` | ✅ |
| `DeepseekV32NaiveMoe.down_proj` | `torch.empty(n_experts, hidden, intermediate)` | ✅ |

**Note:** The MoE classes (`DeepseekV32TopkRouter`, `DeepseekV32NaiveMoe`, `DeepseekV32MoE`) inherit directly from the corresponding DeepSeek V3 classes. This ensures identical behavior for FSDP/ZeRO-3 sharding.

The `_init_weights()` method properly initializes these tensors:
- RMSNorm weights → initialized to 1s
- Router weights → initialized with normal distribution
- Router bias buffer → initialized to 0s (already zero from constructor)
- Expert tensors → initialized with normal distribution

#### Verifying Meta Device Support

```python
import torch
from transformers import DeepseekV32ForCausalLM, DeepseekV32Config

config = DeepseekV32Config(
    hidden_size=64, num_hidden_layers=2, # ... tiny config for testing
)

# Create on meta device
with torch.device("meta"):
    model = DeepseekV32ForCausalLM(config)

# Verify all parameters are on meta device
for name, param in model.named_parameters():
    assert param.device.type == "meta", f"{name} not on meta device"

# Verify all buffers are on meta device
for name, buffer in model.named_buffers():
    assert buffer.device.type == "meta", f"{name} not on meta device"
```

### Gradient Checkpointing

Gradient checkpointing is fully supported:

```python
from transformers import TrainingArguments

args = TrainingArguments(
    gradient_checkpointing=True,
)
```

**Implementation details:**
- Expert routing decisions are computed under `torch.no_grad()` ensuring determinism during recomputation
- Works correctly with both FSDP and ZeRO-3

### Multi-GPU Deployment Modes

| Mode | Use Case | How to Enable |
|------|----------|---------------|
| **FSDP** | Training with parameter sharding | `fsdp="full_shard"` in TrainingArguments |
| **DeepSpeed ZeRO-3** | Training with parameter partitioning | `deepspeed="ds_config.json"` |
| **DDP** | Training with full model replication | Standard PyTorch DDP |
| **Single GPU** | Development/testing | Default (no distributed setup) |

**Note:** `dispatch_model` (Accelerate's device_map) is for **inference only**, not training. For training, use FSDP or ZeRO-3.

---

## Compatibility

- **PyTorch**: >= 2.4.0
- **CUDA**: Tested on H100/H200
- **PEFT**: Compatible (target indexer modules for LoRA)
- **DeepSpeed ZeRO-3**: Compatible
- **FSDP**: Compatible

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

