# PEFT/LoRA reference

This doc outlines how to fine-tune a model with parameter-efficient fine-tuning. This trains small adapter weights instead of the full model. Only adapter parameters are updated — base model weights are frozen. Checkpoints are small (adapter weights only).

```bash
pip install -U peft
```

## Basic LoRA setup

```python
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", torch_dtype="auto")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,              # rank — higher = more capacity, more memory
    lora_alpha=32,    # scaling factor (effective scale = lora_alpha / r)
    lora_dropout=0.1,
    inference_mode=False,
)

model.add_adapter(lora_config)
```

Default target modules are predefined for common models (Llama, Gemma, Qwen target `q_proj`, `v_proj`, etc.). Override explicitly if needed:

```python
LoraConfig(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
# or regex:
LoraConfig(target_modules=[r".*.attn.*"])
```

To fully fine-tune specific layers alongside adapters (e.g., the language model head):

```python
LoraConfig(modules_to_save=["lm_head"])
```

## Training with Trainer

Pass the adapter-wrapped model directly — Trainer detects adapter parameters and updates only those:

```python
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,   # base model with adapter attached
    args=TrainingArguments(output_dir="./output", num_train_epochs=3, ...),
    train_dataset=dataset,
)
trainer.train()
model.save_pretrained("./my-adapter")   # saves adapter weights only
```

Trainer saves only `adapter_model.safetensors` + `adapter_config.json` per checkpoint — base model not included.

## QLoRA (4-bit base model + LoRA)

Combine bitsandbytes 4-bit quantization with LoRA for maximum memory efficiency. The base model is quantized (frozen); only LoRA adapters are trainable:

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False                  # required with gradient_checkpointing
model.enable_input_require_grads()              # required for adapter gradients with frozen base

lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16)
model.add_adapter(lora_config)
```

## Loading and inference

Load adapter from Hub or disk (two ways):

```python
# Auto-detect from adapter_config.json (loads base model automatically)
model = AutoModelForCausalLM.from_pretrained("user/my-lora-adapter")

# Load onto an existing base model
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
model.load_adapter("user/my-lora-adapter")
```

## Multiple adapters

```python
model.add_adapter(LoraConfig(r=8), adapter_name="adapter_1")
model.add_adapter(LoraConfig(r=16), adapter_name="adapter_2")

model.set_adapter("adapter_2")      # activate one
model.enable_adapters()             # activate all
model.disable_adapters()            # disable all (base model only)
model.delete_adapter("adapter_1")   # free memory
```

## Hotswapping (avoid torch.compile recompilation)

Hotswapping replaces adapter weights in-place without triggering recompilation:

```python
model.load_adapter(adapter_path_1)
# ... use adapter 1 ...
model.load_adapter(adapter_path_2, hotswap=True, adapter_name="default")
# ... use adapter 2 — no recompile ...
```

With `torch.compile`, call `enable_peft_hotswap` before compiling:

```python
model.enable_peft_hotswap(target_rank=16)   # set to highest rank across all adapters
model.load_adapter(adapter_path_1, adapter_name="default")
model = torch.compile(model)
# Later:
model.load_adapter(adapter_path_2, adapter_name="default", hotswap=False)
```

Gotcha: Load the adapter with the widest layer coverage first — hotswapping to an adapter that targets more layers than the original triggers recompilation.

## Distributed training

- **ZeRO-3**: Trainer passes `exclude_frozen_parameters=True` automatically — only adapter weights saved in checkpoints.
- **FSDP**: Trainer updates the auto-wrap policy for LoRA layers; for QLoRA it also adjusts the mixed-precision policy.

No manual changes needed for either — Trainer detects the PEFT model and handles it.
