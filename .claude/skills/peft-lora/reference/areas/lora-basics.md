# LoRA Basics (Low-Rank Adaptation)

## Contents
- [Scope](#scope)
- [Minimum questions to ask](#minimum-questions-to-ask)
- [How LoRA works](#how-lora-works)
- [Quickstarts](#quickstarts)
  - [1) Basic LoRA setup for causal LM](#1-basic-lora-setup-for-causal-lm)
  - [2) LoRA for sequence classification](#2-lora-for-sequence-classification)
  - [3) LoRA with automatic target module detection](#3-lora-with-automatic-target-module-detection)
  - [4) Print trainable parameters](#4-print-trainable-parameters)
- [Knobs that matter (3–8)](#knobs-that-matter-38)
- [Pitfalls & fixes](#pitfalls--fixes)

---

## Scope

Use this page when the user wants to:
- Understand LoRA fundamentals
- Set up their first LoRA configuration
- Choose appropriate `r`, `lora_alpha`, and `target_modules`
- Apply LoRA to a pretrained model

---

## Minimum questions to ask

Ask only what you need to produce a runnable snippet (0–5 questions):
1) **Task** (causal LM, classification, seq2seq, etc.)
2) **Model id or local path** (and `revision` if pinned)
3) **Available VRAM** (determines if QLoRA is needed)
4) **Backend + device** (PyTorch; CPU/CUDA/MPS)
5) If blocked: **full traceback + exact versions**

---

## How LoRA works

LoRA (Low-Rank Adaptation) freezes the pretrained model weights and injects trainable rank decomposition matrices into specified layers. Instead of updating a weight matrix `W`, LoRA learns two smaller matrices `A` and `B` such that:

```
W' = W + BA
```

Where:
- `W` is the original frozen weight (e.g., 4096 × 4096)
- `B` has shape (4096 × r) and `A` has shape (r × 4096)
- `r` (rank) is typically 8-64, making trainable parameters << original

**Key insight**: The update `BA` captures task-specific adaptations with minimal parameters.

---

## Quickstarts

### 1. Basic LoRA setup for causal LM

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configure LoRA
lora_config = LoraConfig(
    r=16,                           # Rank of the low-rank matrices
    lora_alpha=32,                  # Scaling factor (effective_alpha = lora_alpha/r)
    lora_dropout=0.05,              # Dropout for LoRA layers
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    task_type=TaskType.CAUSAL_LM,   # Important for proper head handling
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: X || all params: Y || trainable%: Z%
```

---

### 2. LoRA for sequence classification

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model_id = "distilbert/distilbert-base-uncased"

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],  # DistilBERT uses different names
    modules_to_save=["classifier"],      # Train the classification head fully
    task_type=TaskType.SEQ_CLS,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

**Note**: `modules_to_save` ensures the randomly initialized classifier head is trainable and saved with the adapter.

---

### 3. LoRA with automatic target module detection

For many models, you can use `"all-linear"` to automatically target all linear layers:

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",  # Auto-detect all linear layers
    task_type=TaskType.CAUSAL_LM,
)
```

Or let PEFT infer appropriate modules:

```python
# Omit target_modules to use PEFT's default inference
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
)
```

---

### 4. Print trainable parameters

Always verify your configuration worked:

```python
model = get_peft_model(model, lora_config)

# Method 1: Built-in method
model.print_trainable_parameters()

# Method 2: Manual calculation
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")
```

Expected output for a 7B model with LoRA:
```
trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
```

---

## Knobs that matter (3–8)

Prioritize these knobs:

1) **`r` (rank)**
   - Higher = more expressive but more parameters
   - Start with 8-16, increase if underfitting
   
2) **`lora_alpha`**
   - Scaling factor: effective learning = lora_alpha / r
   - Common: 2× the rank (if r=16, use lora_alpha=32)
   
3) **`target_modules`**
   - Which layers to adapt
   - Common patterns:
     - LLMs: `["q_proj", "v_proj"]` or `["q_proj", "k_proj", "v_proj", "o_proj"]`
     - All attention + MLP: `"all-linear"`
   
4) **`task_type`**
   - Must match your model head
   - `CAUSAL_LM`, `SEQ_CLS`, `SEQ_2_SEQ_LM`, `TOKEN_CLS`, `QUESTION_ANS`, `FEATURE_EXTRACTION`
   
5) **`modules_to_save`**
   - Fully train these layers (not LoRA)
   - Use for: classifier heads, newly added tokens, embedding layers
   
6) **`lora_dropout`**
   - Regularization for LoRA layers
   - 0.05-0.1 is typical; 0.0 for inference

7) **`bias`**
   - `"none"` (default): Don't train biases
   - `"all"`: Train all biases
   - `"lora_only"`: Train only LoRA layer biases

---

## Pitfalls & fixes

- **Wrong target_modules for model architecture**
  - Different models use different layer names
  - Llama/Mistral: `q_proj`, `v_proj`, `k_proj`, `o_proj`
  - GPT-2: `c_attn`, `c_proj`
  - BERT: `query`, `value`, `key`
  - **Fix**: Use `"all-linear"` or inspect model with `print(model)`

- **Classification head not training**
  - Add `modules_to_save=["classifier"]` (or appropriate head name)
  - Ensure `task_type=TaskType.SEQ_CLS` is set

- **OOM even with LoRA**
  - LoRA reduces trainable params but still loads full model
  - **Fix**: Use QLoRA (4-bit quantization) - see `qlora-quantization.md`

- **"LoRA layers not found" / empty adapter**
  - `target_modules` didn't match any layers
  - **Fix**: Print model architecture and check layer names

- **Merged model behaves differently**
  - Merging with quantization can cause precision loss
  - **Fix**: Compare outputs before/after merge on fixed inputs

- **Forgetting to set eval mode**
  - LoRA dropout applies during training
  - **Fix**: Call `model.eval()` before inference
