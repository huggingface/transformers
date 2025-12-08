# DeepSeek V3.2 Training Guide

This guide covers training DeepSeek V3.2 models using Modal GPUs with the HuggingFace Transformers implementation.

## Overview

DeepSeek V3.2 extends DeepSeek V3 with a **Lightning Indexer** for sparse attention. The indexer selects top-k tokens per query, reducing attention complexity from O(L²) to O(L×k).

### Key Architecture Components

1. **Multi-head Latent Attention (MLA)**: Low-rank KV projections for memory efficiency
2. **Lightning Indexer**: Selects important tokens for sparse attention
3. **Mixture of Experts (MoE)**: Sparse expert routing for efficient scaling

## Training Modes

### 1. Dense Mode (Recommended for SFT with Short Contexts)

Use dense attention when:
- Context length ≤ 4K tokens
- You want to avoid indexer complexity
- Training from scratch or fine-tuning

```python
from transformers import DeepseekV32Config, DeepseekV32ForCausalLM

config = DeepseekV32Config(
    use_sparse_attention=False,  # Disable sparse attention
    # ... other config options
)
model = DeepseekV32ForCausalLM(config)
```

### 2. Sparse Mode with Frozen Indexer (Recommended for Long-Context SFT)

Use sparse attention with frozen indexer when:
- Context length > 4K tokens
- Using pre-trained indexer weights
- You want the memory benefits of sparse attention without indexer training complexity

```python
from transformers import DeepseekV32ForCausalLM

model = DeepseekV32ForCausalLM.from_pretrained("path/to/checkpoint")

# Freeze indexer weights
for name, param in model.named_parameters():
    if "indexer" in name:
        param.requires_grad = False

# Verify indexer is frozen
indexer_params = sum(p.numel() for n, p in model.named_parameters() if "indexer" in n)
trainable_indexer = sum(p.numel() for n, p in model.named_parameters() if "indexer" in n and p.requires_grad)
print(f"Indexer parameters: {indexer_params:,} (trainable: {trainable_indexer:,})")
```

## Running Tests on Modal

### Prerequisites

1. Install Modal and authenticate:
```bash
pip install modal
modal setup  # Authenticate to 'fairies' workspace
```

2. Ensure the transformers fork is pushed to GitHub:
```bash
cd /path/to/transformers
git push origin shuyingl/deepseek-v3.2-test
```

### Test Commands

**Quick validation (tiny config):**
```bash
modal run scripts/modal_test_deepseek_v32.py --config tiny
```

**Standard test (small config):**
```bash
modal run scripts/modal_test_deepseek_v32.py --config small
```

**Larger test (medium config, more GPU memory):**
```bash
modal run scripts/modal_test_deepseek_v32.py --config medium
```

**SFT test with frozen indexer:**
```bash
modal run scripts/modal_test_deepseek_v32.py --sft --config small
```

**Test official checkpoint (requires HF access):**
```bash
modal run scripts/modal_test_deepseek_v32.py --checkpoint deepseek-ai/DeepSeek-V3.2-Exp
```

### What the Tests Verify

1. **Forward Pass**: Model produces valid logits without NaN/Inf
2. **Backward Pass**: Gradients flow correctly through all parameters
3. **Loss Decreases**: Model learns over 20 training steps

## SFT Training Setup

### Option A: Use research-infra with Custom Model

1. Create a training config that uses the DeepSeek V3.2 model:

```python
# In research-infra repo
from transformers import DeepseekV32ForCausalLM, DeepseekV32Config

def load_model_for_sft(model_path: str, freeze_indexer: bool = True):
    """Load DeepSeek V3.2 for SFT training."""
    model = DeepseekV32ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if freeze_indexer:
        for name, param in model.named_parameters():
            if "indexer" in name:
                param.requires_grad = False

    return model
```

2. Modify `sft/orchestrator_sft_training.py` to use the custom model loader.

### Option B: Dense Mode for Simplicity

For SFT with contexts ≤ 4K tokens, use dense mode to avoid indexer entirely:

```python
config = DeepseekV32Config.from_pretrained(model_path)
config.use_sparse_attention = False  # Switch to dense attention

model = DeepseekV32ForCausalLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.bfloat16,
)
```

## Performance Considerations

### Sparse vs Dense Attention

| Context Length | Recommendation | Reason |
|---------------|----------------|--------|
| ≤ 2K tokens | Dense | Indexer overhead > attention savings |
| 2K-4K tokens | Either | Similar performance |
| 4K-16K tokens | Sparse | Significant memory savings |
| > 16K tokens | Sparse (required) | Dense attention OOM |

### Memory Usage

- **Dense attention**: O(L²) memory for attention scores
- **Sparse attention**: O(L×k) memory where k = index_topk (default 2048)

For a 32K context with sparse attention (k=2048):
- Dense: 32K × 32K = 1B attention elements
- Sparse: 32K × 2K = 64M attention elements (16x reduction)

## Indexer Training with KL-Divergence Loss

Following the DeepSeek V3.2 technical report, the indexer can be trained using KL-divergence loss to align its predictions with full attention patterns.

### When to Train the Indexer

| Scenario | Recommendation |
|----------|----------------|
| Fine-tuning data similar to pretraining | Frozen indexer is fine |
| New domain / significantly different data | Consider indexer warm-up |
| Noticing degraded long-context performance | Train indexer |

### Tech Report Training Procedure

From the DeepSeek V3.2 technical report (arXiv 2512.02556):

| Parameter | Value |
|-----------|-------|
| Steps | 1,000 |
| Batch size | 16 sequences |
| Sequence length | 128K tokens |
| Learning rate | 1e-3 |
| What's frozen | All model parameters except indexer |

### Memory-Efficient API

The model provides a **memory-efficient** API that computes the KL target inside the forward pass, avoiding storage of full attention matrices:

```python
from transformers import DeepseekV32ForCausalLM
from transformers.models.deepseek_v32.modeling_deepseek_v32 import compute_indexer_kl_loss

model = DeepseekV32ForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V3.2")

# Freeze all parameters except indexer
for name, param in model.named_parameters():
    if "indexer" not in name:
        param.requires_grad = False

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3  # From tech report
)

# Training loop
for batch in dataloader:
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        output_indexer_scores=True,      # Return raw I_{t,s} scores
        output_indexer_kl_target=True,   # Return KL target p_{t,:}
    )

    # Compute KL loss using helper function
    kl_loss = compute_indexer_kl_loss(
        outputs.indexer_scores,
        outputs.indexer_kl_targets,
    )

    kl_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### KL Loss Formula

From the tech report:
```
L_I = sum_t D_KL(p_{t,:} || Softmax(I_{t,:}))
```

Where:
- `I_{t,s}` = indexer's raw output scores (returned by `output_indexer_scores=True`)
- `p_{t,:}` = target distribution (returned by `output_indexer_kl_target=True`)

The target `p_{t,:}` is computed as:
1. Full attention scores `[batch, heads, seq, seq]`
2. Sum across heads → `[batch, seq, seq]`
3. L1-normalize along sequence dimension

### Using with PEFT (LoRA)

The indexer has ~3.66B parameters. You can use PEFT to add LoRA for more efficient training:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "indexer.q_b_proj",
        "indexer.k_proj",
        "indexer.weight_proj",
    ],
    lora_dropout=0.0,
)

model = get_peft_model(model, lora_config)
```

Note: Full indexer training (~3.66B params) is only ~0.5% of the total 671B model, so LoRA overhead may not provide significant memory savings. Direct training is often simpler.

### SFT Training Sequence with Indexer Update

For best results when fine-tuning on a significantly different domain:

1. **Stage 1: Indexer Warm-up** (optional but recommended)
   - Freeze all except indexer
   - Train with KL loss only
   - ~1000 steps, LR=1e-3
   - Can use dense attention for accurate KL targets

2. **Stage 2: Main Model SFT**
   - Freeze indexer (now aligned)
   - Train main model with LoRA or full fine-tuning
   - Use sparse attention mode

### Original Multi-Stage Approach (Alternative)

The original paper describes a different multi-stage approach:

1. **Stage 1: Dense Warm-up**
   - Train with `use_sparse_attention=False`
   - This creates good hidden representations for indexer training

2. **Stage 2: Indexer Training**
   - Enable `detach_indexer_input=True` to train indexer separately
   - Use KL-divergence loss between sparse and dense attention outputs
   - Freeze main model, only train indexer

3. **Stage 3: Joint Fine-tuning**
   - Enable sparse attention with trained indexer
   - Fine-tune entire model (or freeze indexer for SFT)

## Troubleshooting

### Common Issues

1. **`fast-hadamard-transform` not installed**
   ```
   pip install fast-hadamard-transform
   ```
   The model will fall back to pure PyTorch if not installed, but it's slower.

2. **OOM on single GPU**
   - Use smaller batch size
   - Enable gradient checkpointing: `model.gradient_checkpointing_enable()`
   - Use `device_map="auto"` for multi-GPU

3. **NaN in loss**
   - Check input token IDs are within vocab_size
   - Ensure proper attention mask
   - Try lower learning rate

### Verifying Installation

```python
from transformers import DeepseekV32Config, DeepseekV32ForCausalLM
import torch

# Quick sanity check
config = DeepseekV32Config(
    vocab_size=100,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    use_sparse_attention=True,
)
model = DeepseekV32ForCausalLM(config)
x = torch.randint(0, 100, (1, 32))
out = model(x)
print(f"Output shape: {out.logits.shape}")  # Should be [1, 32, 100]
```

## Files Reference

- **Model implementation**: `src/transformers/models/deepseek_v32/modular_deepseek_v32.py`
- **Config**: `src/transformers/models/deepseek_v32/configuration_deepseek_v32.py`
- **Local tests**: `scripts/test_deepseek_v32.py`
- **Modal tests**: `scripts/modal_test_deepseek_v32.py`
- **Unit tests**: `tests/models/deepseek_v32/test_modeling_deepseek_v32.py`
