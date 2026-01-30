# Adapter Operations (Load, Save, Merge, Switch)

## Contents
- [Scope](#scope)
- [Minimum questions to ask](#minimum-questions-to-ask)
- [Quickstarts](#quickstarts)
  - [1) Save a trained adapter](#1-save-a-trained-adapter)
  - [2) Load an adapter for inference](#2-load-an-adapter-for-inference)
  - [3) Merge adapter into base model](#3-merge-adapter-into-base-model)
  - [4) Load multiple adapters](#4-load-multiple-adapters)
  - [5) Switch between adapters](#5-switch-between-adapters)
  - [6) Combine adapters with weights](#6-combine-adapters-with-weights)
  - [7) Push adapter to Hub](#7-push-adapter-to-hub)
- [Knobs that matter (3–8)](#knobs-that-matter-38)
- [Pitfalls & fixes](#pitfalls--fixes)

---

## Scope

Use this page when the user wants to:
- Save or load trained LoRA adapters
- Merge adapters into base model
- Work with multiple adapters
- Push/pull adapters from Hugging Face Hub
- Understand adapter file structure

---

## Minimum questions to ask

Ask only what you need (0–5 questions):
1) **Operation**: save, load, merge, or multi-adapter?
2) **Adapter path**: local directory or Hub repo
3) **Base model**: model id (must match adapter's base)
4) **Merge goal**: inference-only or save merged model?
5) If blocked: **full traceback + adapter_config.json contents**

---

## Quickstarts

### 1. Save a trained adapter

After training, save only the adapter weights (not full model):

```python
# After training
model.save_pretrained("./my-lora-adapter")

# This creates:
# ./my-lora-adapter/
#   ├── adapter_config.json    # LoRA configuration
#   ├── adapter_model.safetensors  # Adapter weights (~10-50 MB)
#   └── README.md (optional)
```

**Important**: This saves ONLY the adapter (a few MB), not the base model (several GB).

To also save the tokenizer:
```python
model.save_pretrained("./my-lora-adapter")
tokenizer.save_pretrained("./my-lora-adapter")
```

---

### 2. Load an adapter for inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model first
base_model_id = "meta-llama/Llama-3.1-8B"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Load adapter on top
adapter_path = "./my-lora-adapter"  # or "username/adapter-repo" from Hub
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# Inference
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

### 3. Merge adapter into base model

Merging creates a standalone model without adapter overhead:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base + adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./my-lora-adapter")

# Merge adapter weights into base model
merged_model = model.merge_and_unload()

# Save merged model (full size, no adapter dependency)
merged_model.save_pretrained("./merged-model")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.save_pretrained("./merged-model")
```

**When to merge**:
- Serving without PEFT dependency
- Faster inference (no adapter overhead)
- Creating a standalone checkpoint

**When NOT to merge**:
- Using multiple adapters
- Quantized models (can cause precision loss)
- Need to switch adapters dynamically

---

### 4. Load multiple adapters

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load first adapter
model = PeftModel.from_pretrained(
    base_model,
    "./adapter-style",
    adapter_name="style",  # Give it a name
)

# Load additional adapter
model.load_adapter("./adapter-code", adapter_name="code")

# List available adapters
print(model.peft_config.keys())  # Output: dict_keys(['style', 'code'])
```

---

### 5. Switch between adapters

```python
# Continue from multi-adapter setup

# Set active adapter
model.set_adapter("style")
output_style = model.generate(...)

model.set_adapter("code")
output_code = model.generate(...)

# Disable adapter (use base model)
model.disable_adapter_layers()
output_base = model.generate(...)

# Re-enable adapters
model.enable_adapter_layers()
```

---

### 6. Combine adapters with weights

Use multiple adapters simultaneously with weighted combination:

```python
# Activate multiple adapters at once
model.set_adapter(["style", "code"])

# With custom weights (add_weighted_adapter)
model.add_weighted_adapter(
    adapters=["style", "code"],
    weights=[0.7, 0.3],  # 70% style, 30% code
    adapter_name="combined",
)
model.set_adapter("combined")
```

**Alternative: Linear combination**
```python
from peft import add_weighted_adapter

# Create new adapter from combination
add_weighted_adapter(
    model,
    adapters=["style", "code"],
    weights=[0.5, 0.5],
    combination_type="linear",  # or "cat", "ties", etc.
    adapter_name="mixed",
)
```

---

### 7. Push adapter to Hub

```python
from huggingface_hub import login

# Authenticate
login()  # or login(token="your_token")

# Push adapter
model.push_to_hub("your-username/my-lora-adapter")

# With additional files
tokenizer.push_to_hub("your-username/my-lora-adapter")
```

Loading from Hub:
```python
from peft import PeftModel

model = PeftModel.from_pretrained(
    base_model,
    "your-username/my-lora-adapter",  # Hub path
)
```

---

## Knobs that matter (3–8)

1) **`adapter_name`**
   - Name for identifying adapter when loading multiple
   - Default: `"default"`

2) **`merge_and_unload()` vs keeping adapter**
   - Merge: faster inference, no PEFT dependency, but loses flexibility
   - Keep: can switch adapters, required for quantized models

3) **`safe_serialization`**
   - `True` (default): saves as `.safetensors` (recommended)
   - `False`: saves as `.bin` (legacy)

4) **Base model compatibility**
   - Adapter MUST match base model architecture
   - Check `adapter_config.json` → `base_model_name_or_path`

5) **Revision pinning**
   - Pin base model revision for reproducibility
   - Adapter + base model versions should match

6) **`low_cpu_mem_usage`**
   - Speed up loading for large adapters
   ```python
   PeftModel.from_pretrained(base, adapter, low_cpu_mem_usage=True)
   ```

---

## Pitfalls & fixes

- **"Base model mismatch" or wrong outputs**
  - Adapter trained on different base model version
  - **Fix**: Check `adapter_config.json` and use matching base model

- **Merged quantized model has bad outputs**
  - Merging with 4-bit/8-bit causes precision loss
  - **Fix**: Keep adapter separate or merge before quantization:
  ```python
  # Load in fp16 for merging
  base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
  merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
  # Then quantize the merged model if needed
  ```

- **"RuntimeError: Expected all tensors on same device"**
  - Adapter and base on different devices
  - **Fix**: Use `device_map="auto"` for both or manually move

- **Adapter not applying / unchanged outputs**
  - Adapter might be disabled
  - **Fix**: Check adapter state:
  ```python
  from peft import get_model_status
  print(get_model_status(model))  # Shows active adapters, merged status
  ```

- **OOM when loading adapter**
  - Full model + adapter exceeds VRAM
  - **Fix**: Use quantized base model or `low_cpu_mem_usage=True`

- **Multiple adapters conflict**
  - Both adapters target same layers differently
  - **Fix**: Use weighted combination or train single multi-task adapter

- **"ValueError: Cannot add a new adapter when adapter is merged"**
  - Called `merge_and_unload()` then tried to add adapter
  - **Fix**: Reload base model, or use `merge_adapter()` (doesn't unload):
  ```python
  # Merge in place (can still add more adapters)
  model.merge_adapter()
  # vs merge_and_unload() which removes adapter infrastructure
  ```

---

## Adapter file structure

Understanding what's saved:

```
my-lora-adapter/
├── adapter_config.json      # LoRA config (r, alpha, target_modules, etc.)
├── adapter_model.safetensors  # Trained LoRA weights
└── README.md                # Auto-generated model card

# adapter_config.json example:
{
  "base_model_name_or_path": "meta-llama/Llama-3.1-8B",
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "v_proj"],
  "task_type": "CAUSAL_LM",
  "peft_type": "LORA"
}
```

The adapter is typically **very small** compared to the base model:
- 7B base model: ~14 GB
- Typical LoRA adapter: 10-50 MB
