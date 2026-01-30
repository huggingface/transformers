# Troubleshooting (PEFT Errors, Wrong Outputs, Common Issues)

## Contents
- [Scope](#scope)
- [Minimum questions to ask](#minimum-questions-to-ask)
- [Decision guide: classify the failure](#decision-guide-classify-the-failure)
- [Quickstarts](#quickstarts)
  - [1) "ValueError: Attempting to unscale FP16 gradients"](#1-valueerror-attempting-to-unscale-fp16-gradients)
  - [2) Loaded adapter produces wrong/random outputs](#2-loaded-adapter-produces-wrongrandom-outputs)
  - [3) "Target modules not found" / empty adapter](#3-target-modules-not-found--empty-adapter)
  - [4) OOM even with LoRA](#4-oom-even-with-lora)
  - [5) Loss is None / labels not passed](#5-loss-is-none--labels-not-passed)
  - [6) Adapter loading version mismatch](#6-adapter-loading-version-mismatch)
  - [7) Check model and layer status](#7-check-model-and-layer-status)
- [Pitfalls & fixes (quick reference)](#pitfalls--fixes-quick-reference)
- [Triage flow (repeatable checklist)](#triage-flow-repeatable-checklist)

---

## Scope

Use this page when the user is **blocked** by:
- Exceptions during PEFT training/inference
- Wrong outputs from trained adapters
- Memory issues despite using PEFT
- Adapter loading/saving failures
- Dtype mismatches

---

## Minimum questions to ask

Ask only what you need (0–5 questions). If already provided, don't re-ask:
1) **Exact error**: full traceback
2) **Minimal repro**: smallest code that fails
3) **Versions**: `peft`, `transformers`, `torch`, `bitsandbytes` versions
4) **Model + adapter config**: model id, LoraConfig settings
5) **Hardware**: CUDA/CPU, VRAM if memory-related

---

## Decision guide: classify the failure

Classify before fixing:

1) **Dtype/precision issues**
   - FP16 gradient errors, mixed precision conflicts
   - NaN loss, overflow

2) **Configuration issues**
   - Wrong target_modules, task_type mismatch
   - Adapter not applying

3) **Loading issues**
   - Version mismatch, missing files
   - Base model incompatibility

4) **Training issues**
   - Loss is None, labels not used
   - Training but no improvement

5) **Memory issues**
   - OOM despite LoRA/QLoRA
   - Slow due to wrong settings

---

## Quickstarts

### 1. "ValueError: Attempting to unscale FP16 gradients"

**Symptom**: Error during training with fp16 and quantized model.

**Cause**: Model loaded in fp16, used with AMP, but trainable weights should be fp32.

**Fix**: Cast trainable parameters:

```python
from peft import get_peft_model, cast_mixed_precision_params

# After creating PEFT model
model = get_peft_model(model, lora_config)

# Cast trainable params to appropriate dtype
cast_mixed_precision_params(model, dtype=torch.float16)

# Now training with fp16=True works
trainer = Trainer(model=model, args=TrainingArguments(fp16=True, ...))
```

**Alternative** (manual):
```python
for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.float()
```

---

### 2. Loaded adapter produces wrong/random outputs

**Symptom**: Adapter loads without error but outputs are garbage.

**Possible causes and fixes**:

**A) Wrong base model**
```python
# Check adapter_config.json for expected base
import json
with open("./my-adapter/adapter_config.json") as f:
    config = json.load(f)
print(config["base_model_name_or_path"])  # Must match!
```

**B) Forgot to set eval mode**
```python
model = PeftModel.from_pretrained(base, adapter)
model.eval()  # IMPORTANT: disable dropout
```

**C) Adapter not active**
```python
from peft import get_model_status
print(get_model_status(model))
# Check: enabled=True, active_adapters=['default']
```

**D) Random deviations**
```python
# Set deterministic mode
import torch
torch.manual_seed(42)
model.eval()
with torch.inference_mode():
    outputs = model.generate(...)
```

---

### 3. "Target modules not found" / empty adapter

**Symptom**: LoRA not applied, 0 trainable parameters.

**Cause**: `target_modules` don't match model's layer names.

**Fix**: Inspect model architecture:

```python
# Print all named modules
for name, module in model.named_modules():
    print(name, type(module).__name__)

# Common patterns:
# Llama/Mistral: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
# GPT-2: c_attn, c_proj, c_fc
# BERT: query, key, value, dense
# DistilBERT: q_lin, k_lin, v_lin, out_lin
```

**Safe approach** - use auto-detection:
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",  # Auto-detect all Linear layers
    task_type=TaskType.CAUSAL_LM,
)
```

---

### 4. OOM even with LoRA

**Symptom**: CUDA out of memory during training.

**LoRA reduces trainable params, NOT forward pass memory.**

**Fixes (in order)**:

```python
# 1. Use QLoRA (4-bit quantization)
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 2. Reduce batch size + increase accumulation
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Minimum
    gradient_accumulation_steps=16,  # Effective batch = 16
)

# 3. Enable gradient checkpointing (auto with QLoRA)
model.gradient_checkpointing_enable()

# 4. Reduce sequence length
max_length = 256  # vs 512 or 1024

# 5. Use paged optimizer
training_args = TrainingArguments(
    optim="paged_adamw_8bit",
)
```

---

### 5. Loss is None / labels not passed

**Symptom**: Training runs but loss shows `None` or model doesn't improve.

**Cause**: Labels not passed to model forward.

**Fixes**:

```python
# For causal LM: labels = input_ids (shifted internally)
def tokenize(examples):
    tokenized = tokenizer(examples["text"], truncation=True, max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Labels = inputs
    return tokenized

# For classification: ensure labels column exists
dataset = dataset.rename_column("label", "labels")

# Check during forward
outputs = model(input_ids=input_ids, labels=labels)
print(outputs.loss)  # Should be a tensor, not None
```

---

### 6. Adapter loading version mismatch

**Symptom**: `TypeError: LoraConfig.__init__() got an unexpected keyword argument`

**Cause**: Adapter saved with newer PEFT than installed version.

**Fix**:
```bash
# Upgrade PEFT
pip install -U peft

# Or from source for latest
pip install git+https://github.com/huggingface/peft
```

**Workaround** (if can't upgrade):
```python
# Edit adapter_config.json and remove the unknown field
# This works if it's a default value, may cause issues otherwise
```

---

### 7. Check model and layer status

Use these to debug adapter state:

```python
from peft import get_model_status, get_layer_status

# Overall model status
status = get_model_status(model)
print(status)
# TunerModelStatus(
#     base_model_type='LlamaForCausalLM',
#     adapter_model_type='LoraModel',
#     trainable_params=4194304,
#     total_params=6738415616,
#     enabled=True,
#     active_adapters=['default'],
#     merged_adapters=[],
# )

# Per-layer status
for layer in get_layer_status(model)[:5]:
    print(layer)
# Shows: name, module_type, enabled, active_adapters, merged_adapters

# Check for "irregular" status (inconsistent state)
if "irregular" in str(status):
    print("WARNING: Model in inconsistent state, reload!")
```

---

## Pitfalls & fixes (quick reference)

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| "Attempting to unscale FP16 gradients" | Trainable params in fp16 with AMP | `cast_mixed_precision_params()` |
| 0 trainable parameters | Wrong target_modules | Use `"all-linear"` or inspect model |
| Loaded model gives garbage | Wrong base model / not in eval mode | Check adapter_config.json, call `.eval()` |
| OOM with LoRA | Forward pass still full size | Use QLoRA, reduce batch/seq length |
| Loss is None | Labels not passed | Add `labels=input_ids` for LM |
| "unexpected keyword argument" | PEFT version mismatch | Upgrade: `pip install -U peft` |
| NaN loss | Learning rate too high / dtype issues | Lower LR, check compute dtype |
| Training slow | Gradient checkpointing off | Enable with `prepare_model_for_kbit_training()` |
| Merged model worse | Quantization precision loss | Merge in fp16, then quantize |
| Multiple adapters conflict | Same layers, different configs | Use weighted combination |

---

## Triage flow (repeatable checklist)

1) **Freeze the environment**
   - Record: peft, transformers, torch, bitsandbytes versions
   - Record: model id, LoraConfig
   - Record: hardware (GPU, VRAM)

2) **Minimize**
   - One model
   - One input
   - One forward/generate call
   - Print shapes, dtypes, devices

3) **Classify**
   - Dtype → Configuration → Loading → Training → Memory

4) **Check adapter state**
   ```python
   from peft import get_model_status
   print(get_model_status(model))
   model.print_trainable_parameters()
   ```

5) **Apply smallest fix**
   - One change at a time
   - Re-run minimal repro

6) **Verify fix worked**
   - Check trainable params > 0
   - Check loss is decreasing
   - Check outputs are sensible

---

## Common version requirements

For most PEFT functionality:
```bash
pip install peft>=0.10.0 transformers>=4.38.0 bitsandbytes>=0.43.0
```

For latest features:
```bash
pip install git+https://github.com/huggingface/peft
pip install git+https://github.com/huggingface/transformers
```

Check installed versions:
```python
import peft, transformers, torch
print(f"peft: {peft.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"torch: {torch.__version__}")
try:
    import bitsandbytes as bnb
    print(f"bitsandbytes: {bnb.__version__}")
except ImportError:
    print("bitsandbytes: not installed")
```
