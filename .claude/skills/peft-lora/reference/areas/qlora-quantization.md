# QLoRA & Quantization (Memory-Efficient Fine-Tuning)

## Contents
- [Scope](#scope)
- [Minimum questions to ask](#minimum-questions-to-ask)
- [How QLoRA works](#how-qlora-works)
- [Quickstarts](#quickstarts)
  - [1) QLoRA 4-bit training setup](#1-qlora-4-bit-training-setup)
  - [2) 8-bit LoRA training](#2-8-bit-lora-training)
  - [3) NF4 vs FP4 quantization](#3-nf4-vs-fp4-quantization)
  - [4) Compute dtype selection](#4-compute-dtype-selection)
- [Knobs that matter (3–8)](#knobs-that-matter-38)
- [Pitfalls & fixes](#pitfalls--fixes)
- [Memory comparison](#memory-comparison)

---

## Scope

Use this page when the user wants to:
- Fine-tune large models on consumer GPUs (8-24GB VRAM)
- Use 4-bit or 8-bit quantization with LoRA
- Understand QLoRA configuration options
- Troubleshoot memory issues during training

---

## Minimum questions to ask

Ask only what you need (0–5 questions):
1) **Model size** (7B, 13B, 70B, etc.)
2) **Available VRAM** (determines 4-bit vs 8-bit)
3) **GPU type** (affects compute dtype choice)
4) **Model id or local path**
5) If blocked: **full OOM traceback + batch size + sequence length**

---

## How QLoRA works

QLoRA combines three techniques:
1. **4-bit NormalFloat (NF4)** quantization for base model weights
2. **Double quantization** to reduce memory from quantization constants
3. **LoRA adapters** trained in higher precision (bf16/fp16)

The base model stays frozen in 4-bit, while LoRA adapters train in 16-bit precision, achieving near-full-precision quality with ~75% less memory.

---

## Quickstarts

### 1. QLoRA 4-bit training setup

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

model_id = "meta-llama/Llama-3.1-8B"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 (recommended)
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute dtype for LoRA
    bnb_4bit_use_double_quant=True,      # Double quantization saves more memory
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# CRITICAL: Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

**Critical**: Always call `prepare_model_for_kbit_training()` before `get_peft_model()` for quantized models.

---

### 2. 8-bit LoRA training

For more precision at slightly higher memory cost:

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

model_id = "meta-llama/Llama-3.1-8B"

# 8-bit config
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
```

---

### 3. NF4 vs FP4 quantization

```python
from transformers import BitsAndBytesConfig
import torch

# NF4 (recommended for most cases)
bnb_config_nf4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4: optimized for normally-distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# FP4 (alternative, sometimes faster)
bnb_config_fp4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",  # Standard 4-bit float
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

**When to use each**:
- **NF4**: Default choice, better for transformer weights (normally distributed)
- **FP4**: Try if NF4 causes issues or for non-standard architectures

---

### 4. Compute dtype selection

```python
import torch
from transformers import BitsAndBytesConfig

# For Ampere+ GPUs (RTX 3000/4000, A100, H100)
bnb_config_bf16 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # Native bf16 support
    bnb_4bit_use_double_quant=True,
)

# For older GPUs (V100, RTX 2000, etc.)
bnb_config_fp16 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,   # Use fp16 instead
    bnb_4bit_use_double_quant=True,
)
```

**Check GPU capability**:
```python
import torch
if torch.cuda.is_bf16_supported():
    print("Use bfloat16")
else:
    print("Use float16")
```

---

## Knobs that matter (3–8)

1) **`load_in_4bit` vs `load_in_8bit`**
   - 4-bit: Maximum memory savings (~50% of 8-bit)
   - 8-bit: Better precision, still significant savings

2) **`bnb_4bit_quant_type`**
   - `"nf4"`: NormalFloat4, optimized for neural network weights
   - `"fp4"`: Standard 4-bit float

3) **`bnb_4bit_compute_dtype`**
   - Dtype for LoRA computations (not storage)
   - `torch.bfloat16`: Preferred for Ampere+ GPUs
   - `torch.float16`: For older GPUs

4) **`bnb_4bit_use_double_quant`**
   - Quantizes quantization constants (saves ~0.4 bits/param)
   - Recommended: `True`

5) **`prepare_model_for_kbit_training()`**
   - REQUIRED for quantized training
   - Handles gradient checkpointing and layer norms

6) **LoRA `r` with quantization**
   - May need higher `r` to compensate for quantization
   - Start with r=16-32 for QLoRA (vs r=8-16 for fp16 LoRA)

7) **Gradient checkpointing**
   - Automatically enabled by `prepare_model_for_kbit_training()`
   - Can disable if VRAM allows: `model.gradient_checkpointing_disable()`

---

## Pitfalls & fixes

- **"ValueError: Attempting to unscale FP16 gradients"**
  - Caused by fp16 model + mixed precision training
  - **Fix**: Cast trainable params to fp32:
  ```python
  from peft import cast_mixed_precision_params
  cast_mixed_precision_params(model, dtype=torch.float16)
  ```

- **Forgot `prepare_model_for_kbit_training()`**
  - Training may fail or produce bad results
  - **Fix**: Always call before `get_peft_model()`:
  ```python
  model = prepare_model_for_kbit_training(model)
  model = get_peft_model(model, lora_config)
  ```

- **OOM even with 4-bit**
  - Reduce batch size first
  - Enable gradient checkpointing (default with prepare_model_for_kbit_training)
  - Reduce sequence length
  - Use gradient accumulation:
  ```python
  training_args = TrainingArguments(
      per_device_train_batch_size=1,
      gradient_accumulation_steps=16,  # Effective batch = 16
  )
  ```

- **Training loss explodes or NaN**
  - Check compute dtype matches GPU capability
  - Try different learning rate (lower for QLoRA, e.g., 1e-4 to 2e-4)
  - Ensure optimizer compatible with 4-bit:
  ```python
  # Use paged AdamW for memory efficiency
  training_args = TrainingArguments(
      optim="paged_adamw_8bit",
  )
  ```

- **bitsandbytes not installed or CUDA issues**
  ```bash
  pip install bitsandbytes>=0.43.0
  ```
  - Ensure CUDA toolkit matches PyTorch CUDA version

- **Model outputs are wrong after quantization**
  - Compare outputs on same inputs with fp16 model
  - Try `bnb_4bit_quant_type="fp4"` if `nf4` causes issues
  - Check if model architecture is supported by bitsandbytes

---

## Memory comparison

Approximate VRAM usage for fine-tuning:

| Model Size | Full FT (fp16) | LoRA (fp16) | LoRA (8-bit) | QLoRA (4-bit) |
|---|---|---|---|---|
| 7B | 28+ GB | 14-16 GB | 10-12 GB | 6-8 GB |
| 13B | 52+ GB | 28-32 GB | 18-22 GB | 10-14 GB |
| 70B | 280+ GB | 140+ GB | 80-100 GB | 40-50 GB |

*Values vary based on sequence length, batch size, and gradient accumulation.*

### Recommended GPU per model size (QLoRA):
- **7B**: RTX 3090/4090 (24GB), A10 (24GB)
- **13B**: A100-40GB, 2x RTX 3090
- **70B**: A100-80GB, 4x A100-40GB
