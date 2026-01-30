# Training Integration (Trainer, SFTTrainer, Custom Loops)

## Contents
- [Scope](#scope)
- [Minimum questions to ask](#minimum-questions-to-ask)
- [Decision guide: Trainer vs SFTTrainer vs custom loop](#decision-guide-trainer-vs-sfttrainer-vs-custom-loop)
- [Quickstarts](#quickstarts)
  - [1) PEFT with Transformers Trainer](#1-peft-with-transformers-trainer)
  - [2) QLoRA with TRL SFTTrainer](#2-qlora-with-trl-sfttrainer)
  - [3) Custom training loop with Accelerate](#3-custom-training-loop-with-accelerate)
  - [4) Resume training from checkpoint](#4-resume-training-from-checkpoint)
  - [5) Multi-GPU training with PEFT](#5-multi-gpu-training-with-peft)
- [Knobs that matter (3–8)](#knobs-that-matter-38)
- [Pitfalls & fixes](#pitfalls--fixes)

---

## Scope

Use this page when the user wants to:
- Train LoRA/QLoRA with Transformers Trainer
- Use TRL's SFTTrainer for instruction fine-tuning
- Write custom training loops with PEFT
- Resume training from checkpoints
- Scale PEFT training to multiple GPUs

---

## Minimum questions to ask

Ask only what you need (0–6 questions):
1) **Task** (instruction tuning, classification, causal LM, etc.)
2) **Training framework** (Trainer, SFTTrainer, custom loop)
3) **Model + PEFT config** (which model, LoRA or QLoRA)
4) **Hardware** (single GPU, multi-GPU, VRAM)
5) **Dataset format** (already tokenized, needs preprocessing)
6) If blocked: **full traceback + training args**

---

## Decision guide: Trainer vs SFTTrainer vs custom loop

### Prefer `Trainer` when…
- Standard supervised fine-tuning
- Classification or sequence tasks
- Need full control over preprocessing
- Already familiar with Trainer API

### Prefer `SFTTrainer` (from TRL) when…
- Instruction fine-tuning / chat models
- Want automatic chat template handling
- Using datasets with conversational format
- Want RLHF-ready training setup

### Prefer custom loop when…
- Non-standard objectives (RL, multi-stage, etc.)
- Very custom batching or optimization
- Research experimentation
- Full control over every step

---

## Quickstarts

### 1. PEFT with Transformers Trainer

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# Model setup
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# Training
training_args = TrainingArguments(
    output_dir="./lora-output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
trainer.save_model("./lora-output/final")
```

---

### 2. QLoRA with TRL SFTTrainer

Best for instruction fine-tuning:

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Model with 4-bit quantization
model_id = "meta-llama/Llama-3.1-8B"
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
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# Dataset with instruction format
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train[:1000]")

# SFT Training
training_args = SFTConfig(
    output_dir="./qlora-sft",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    max_seq_length=512,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,  # SFTTrainer handles PEFT integration
)

trainer.train()
trainer.save_model("./qlora-sft/final")
```

---

### 3. Custom training loop with Accelerate

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator

# Setup
accelerator = Accelerator()
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules="all-linear",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# Optimizer (only for trainable params)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-4,
)

# Prepare with Accelerate
model, optimizer = accelerator.prepare(model, optimizer)

# Training loop (simplified)
model.train()
for epoch in range(3):
    for batch in dataloader:  # Your DataLoader
        outputs = model(**batch)
        loss = outputs.loss
        
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

# Save (unwrap from accelerate)
unwrapped = accelerator.unwrap_model(model)
unwrapped.save_pretrained("./custom-lora")
```

---

### 4. Resume training from checkpoint

```python
from transformers import Trainer, TrainingArguments
from peft import PeftModel

# Option 1: Resume with Trainer (automatic)
training_args = TrainingArguments(
    output_dir="./lora-output",
    resume_from_checkpoint=True,  # Auto-find latest checkpoint
    # or: resume_from_checkpoint="./lora-output/checkpoint-500"
)

# Option 2: Manual checkpoint loading
base_model = AutoModelForCausalLM.from_pretrained(model_id)
model = PeftModel.from_pretrained(base_model, "./lora-output/checkpoint-500")
# Continue training with new Trainer instance...
```

---

### 5. Multi-GPU training with PEFT

PEFT works seamlessly with distributed training:

```bash
# With accelerate (recommended)
accelerate config  # Interactive setup
accelerate launch train.py

# With torchrun
torchrun --nproc_per_node 4 train.py
```

**FSDP with PEFT** (for very large models):

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./fsdp-lora",
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
    },
    # ... other args
)
```

---

## Knobs that matter (3–8)

1) **Learning rate**
   - LoRA typically needs higher LR than full fine-tuning
   - Start: 1e-4 to 3e-4 (vs 1e-5 to 5e-5 for full FT)

2) **`gradient_accumulation_steps`**
   - Critical for memory-constrained training
   - Effective batch = per_device_batch × accumulation × num_gpus

3) **`optim`**
   - `"paged_adamw_8bit"`: Memory-efficient for QLoRA
   - `"adamw_torch"`: Standard choice

4) **`fp16` vs `bf16`**
   - `bf16=True`: For Ampere+ GPUs, better numerical stability
   - `fp16=True`: For older GPUs

5) **`max_grad_norm`**
   - Gradient clipping for stability
   - Default: 1.0, can lower to 0.3 if loss spikes

6) **`warmup_ratio` or `warmup_steps`**
   - Helps stability at training start
   - 0.03-0.1 warmup_ratio is common

7) **`save_strategy` and checkpointing**
   - Save frequently (e.g., "steps", save_steps=100)
   - Adapters are small, so saving is cheap

8) **`prepare_model_for_kbit_training()`**
   - REQUIRED for quantized training
   - Handles gradient checkpointing automatically

---

## Pitfalls & fixes

- **"RuntimeError: element 0 of tensors does not require grad"**
  - No trainable parameters (LoRA not applied correctly)
  - **Fix**: Check `model.print_trainable_parameters()` shows > 0

- **Loss is NaN or explodes**
  - Learning rate too high
  - **Fix**: Lower LR (try 1e-4), add gradient clipping:
  ```python
  training_args = TrainingArguments(
      learning_rate=1e-4,
      max_grad_norm=0.3,
  )
  ```

- **OOM during training (even with LoRA)**
  - Batch size too large or sequence too long
  - **Fix**: 
    - Reduce `per_device_train_batch_size` to 1
    - Increase `gradient_accumulation_steps`
    - Use QLoRA (4-bit)
    - Reduce `max_seq_length`

- **Trainer not saving adapter correctly**
  - Using old `save_model()` behavior
  - **Fix**: PEFT auto-integrates; check output has `adapter_config.json`

- **Checkpoints are huge (full model size)**
  - Might be saving full model instead of adapter
  - **Fix**: Ensure you're calling `model.save_pretrained()` on PeftModel

- **"ValueError: You are attempting to train a model on a text generation task"**
  - Task type mismatch in LoRA config
  - **Fix**: Set `task_type=TaskType.CAUSAL_LM` explicitly

- **Multi-GPU: adapters not synced**
  - Should work automatically with DDP/FSDP
  - **Fix**: Ensure `device_map="auto"` or proper FSDP wrapping

- **Training slow despite small trainable params**
  - Full model still does forward pass
  - LoRA reduces trainable params, not forward pass cost
  - **Fix**: Use gradient checkpointing, reduce sequence length

---

## Common training configurations

### Instruction Fine-Tuning (QLoRA, 7B model)
```python
training_args = TrainingArguments(
    output_dir="./instruction-ft",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    bf16=True,  # or fp16=True for older GPUs
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
)
```

### Classification Fine-Tuning (LoRA)
```python
training_args = TrainingArguments(
    output_dir="./classification-ft",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    learning_rate=1e-4,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)
```
