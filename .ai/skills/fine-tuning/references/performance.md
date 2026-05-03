# Performance reference

This doc outlines the performance (training speed and memory-usage) optimizations available for fine-tuning.

## Understanding GPU memory usage

Training a model consumes memory across several categories (approximate bytes per parameter):

| Component | Bytes/param | Notes |
|---|---|---|
| Model weights (mixed precision) | 6 | fp16 copy (2B) for forward/backward + fp32 master copy (4B) for optimizer |
| Adam optimizer states | 8 | fp32 momentum (4B) + fp32 variance (4B) |
| Gradients | 4 | Computed in fp32 during backward |
| Activations | Varies | Grows with batch × seq_len × depth × hidden |

Attention score matrices scale with the **square** of sequence length — the main reason long sequences are expensive.

With 8-bit Adam (`optim="adamw_8bit"`), optimizer states drop from 8B → 2B per parameter.

## Memory optimization techniques

| Technique | TrainingArguments flag | Tradeoff |
|---|---|---|
| Mixed precision (bf16) | `bf16=True` | Negligible accuracy loss; prefer over fp16 on Ampere+ |
| Mixed precision (fp16) | `fp16=True` | Requires loss scaling; use on pre-Ampere |
| TF32 compute | `tf32=True` | ~8× faster matmuls on Ampere at no precision cost; pair with bf16 |
| Flash Attention 2 | `attn_implementation="flash_attention_2"` (model load) | 2–4× faster attention, lower peak memory; requires `pip install flash-attn` |
| SDPA attention | `attn_implementation="sdpa"` (model load) | Faster than eager attention; no extra install; use when flash-attn unavailable |
| Gradient checkpointing | `gradient_checkpointing=True` | ~20% slower; saves activation memory |
| Gradient accumulation | `gradient_accumulation_steps=N` | Simulates N× batch size; no memory cost |
| 8-bit Adam | `optim="adamw_8bit"` | Requires bitsandbytes; ~75% optimizer state reduction |
| Adafactor | `optim="adafactor"` | Stateless; very low memory; slower convergence |
| Eval incremental offload | `eval_accumulation_steps=16` | Prevents eval OOM on large models |
| Dataloader prefetch | `dataloader_prefetch_factor=2` | Prefetch 2 batches per worker; requires `num_workers > 0` |

**Mixed precision gotcha**: Load the model in fp32 initially (or `torch_dtype="auto"`), not in bf16/fp16. Loading in reduced precision leaves no fp32 master copy for the optimizer to update from, making autocast a no-op.

**Gradient checkpointing gotcha**: Set `model.config.use_cache = False` on generative models — KV cache is incompatible with activation recomputation.

**Gradient accumulation + custom loss**: When using a custom loss function, normalize by `num_items_in_batch` (non-padding token count), not by `gradient_accumulation_steps`. Using steps as the divisor gives wrong gradients for variable-length sequences.

## Eval OOM prevention

On large models, accumulating all logits in GPU memory during eval causes OOM. Use both together:

```python
# In TrainingArguments:
eval_accumulation_steps=16   # flush predictions to CPU every 16 batches

# In Trainer:
def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)  # reduce (batch, seq, vocab) → (batch, seq) on GPU
```

## Dataloader tuning

Default `dataloader_num_workers=0` causes the GPU to idle while the CPU loads the next batch:

```python
TrainingArguments(
    dataloader_num_workers=4,
    dataloader_persistent_workers=True,   # keep worker processes alive between epochs
    dataloader_prefetch_factor=2,         # prefetch 2 batches per worker
)
```

`persistent_workers` and `prefetch_factor` require `num_workers > 0`.

## Kernels (Liger)

Liger Kernel provides fused GPU kernels for common transformer ops (RoPE, RMSNorm, SwiGLU, cross-entropy). Reduces memory bandwidth usage and kernel launch overhead.

```bash
pip install liger-kernel
```

```python
TrainingArguments(
    use_liger_kernel=True,
    liger_kernel_config={
        "rope": True,
        "cross_entropy": True,
        "rms_norm": True,
        "swiglu": True,
    }
)
```

Compatible with FlashAttention, FSDP, and DeepSpeed. Available layer options vary by model architecture.

## Kernels (Hub kernels via KernelConfig)

Load custom kernels from the Hub to replace specific model layers:

```python
from transformers import AutoModelForCausalLM, KernelConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    use_kernels=True,
    kernel_config=KernelConfig(
        kernel_mapping={"RMSNorm": "kernels-community/rmsnorm"}
    ),
)
```

## torch.compile

Compiles the forward and backward passes together into fused kernels. Adds a one-time compilation overhead on the first step; subsequent steps are faster.

```python
TrainingArguments(
    torch_compile=True,
    torch_compile_backend="inductor",         # default; best for most training
    torch_compile_mode="reduce-overhead",     # lower Python overhead via CUDA graphs
    # other modes: "default", "max-autotune", "max-autotune-no-cudagraphs"
)
```

Not compatible with all architectures. If training fails on compile, remove `torch_compile=True`.

## NEFTune (instruction fine-tuning regularization)

Adds random noise to token embeddings during training. Only active during training; removed at eval/inference. Consistently improves instruction fine-tuning quality.

```python
TrainingArguments(
    neftune_noise_alpha=5,   # range 5–15; higher = more regularization
)
```
