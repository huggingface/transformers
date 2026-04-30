---
name: fine-tuning
description: Fine-tune a model with the Transformers Trainer API. Use when fine-tuning any model (text, vision, audio, multimodal) using the Trainer class, setting up TrainingArguments, writing custom callbacks or loss functions, optimizing memory or throughput, or setting up distributed training across multiple GPUs.
---

# Fine-Tuning with the Trainer API

## Input

- `<model_id>`: model name or local path
- `<train_dataset>`: a `datasets.Dataset` or any `torch.utils.data.Dataset`
- `<task>`: classification, generation, seq2seq, token classification, etc.

## Core Workflow

### 1. Load model and tokenizer

Use the task-appropriate `AutoModel` class. Load with `dtype="auto"` to avoid doubling memory (avoids upcasting to float32):

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,
    torch_dtype="auto",
)
```

Tokenize with `dataset.map(tokenize_fn, batched=True)`. Use `DataCollatorWithPadding` for dynamic padding (more memory-efficient than padding to max_length upfront). For causal LM use `DataCollatorForLanguageModeling(mlm=False)`.

### 2. Configure TrainingArguments

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,     # effective batch = per_device × n_gpus × accumulation
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    bf16=True,                         # prefer bf16 over fp16 on Ampere+
    gradient_checkpointing=True,       # saves memory, ~20% slower
    dataloader_num_workers=4,          # non-zero prevents GPU stalling on data load
    dataloader_persistent_workers=True,
    eval_strategy="epoch",
    save_strategy="epoch",             # must match eval_strategy if load_best_model_at_end=True
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=50,
    report_to="none",                  # or "wandb", "tensorboard"
)
```

**Checkpointing constraint**: When `load_best_model_at_end=True`:
- `eval_strategy` must not be `"no"`
- `save_strategy` must match `eval_strategy`, with two exceptions:
  - `save_strategy="best"` — saves only on new best metric, most disk-efficient; exempt from matching
  - If using `"steps"`, `save_steps` must be a multiple of `eval_steps`
- `metric_for_best_model` is required

**Resuming constraint**: `save_only_model=True` saves disk space but strips optimizer/scheduler/RNG state, making checkpoint resuming impossible. Leave `False` (default) if you may resume.

### 3. Define compute_metrics (optional but recommended)

```python
import evaluate
import numpy as np

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

To prevent eval OOM on large models, preprocess logits on GPU before accumulation:

```python
def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)  # reduces from (batch, seq, vocab) to (batch, seq)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred  # predictions are already argmax'd
    return metric.compute(predictions=predictions, references=labels)
```

Also set `eval_accumulation_steps=16` in TrainingArguments to offload eval predictions to CPU incrementally.

### 4. Create the Trainer and train

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./final-model")
```

To resume: `trainer.train(resume_from_checkpoint="./output/checkpoint-500")` or `resume_from_checkpoint=True` for the latest.

Add `eval_on_start=True` to `TrainingArguments` to validate the eval pipeline before a long run.

## Common mistakes

- Forgetting `eval_dataset` when `eval_strategy != "no"` — Trainer will raise at the first eval step
- Mismatching `save_strategy` / `eval_strategy` with `load_best_model_at_end=True` — raises `ValueError` at init
- `gradient_checkpointing=True` without `model.config.use_cache = False` on generative models — cache is incompatible with recomputation
- Setting `max_steps` and `num_train_epochs` together — `max_steps` wins
- `dataloader_num_workers=0` (default) — GPU idles during data loading; use 4+ workers

## PEFT / LoRA (parameter-efficient fine-tuning)

To train adapter weights instead of the full model (`pip install -U peft`):

```python
from peft import LoraConfig, TaskType

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
model.add_adapter(LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1))
# Then pass model to Trainer as normal — only adapter params are updated
```

For QLoRA (4-bit base model + LoRA), combine with `BitsAndBytesConfig(load_in_4bit=True)` and call `model.enable_input_require_grads()` before adding the adapter. See `references/peft.md` for the full pattern including multiple adapters, hotswapping, and distributed training.

## Advanced topics

Read the relevant reference file before writing code for any of these:

- **Task recipes** (`references/task-recipes.md`): task-specific model class, tokenization, data collator, and metrics for sequence classification, token classification, causal/masked LM, QA, summarization, translation, image classification, object detection, segmentation, VQA, captioning, ASR, audio classification, video classification
- **Performance** (`references/performance.md`): memory anatomy, Liger kernels, Hub kernels, mixed precision gotchas, NEFTune, torch.compile
- **Customization** (`references/customization.md`): data collators, custom loss, callbacks, subclassing, optimizers/schedulers, hyperparameter search
- **PEFT / LoRA** (`references/peft.md`): LoRA setup, QLoRA, multiple adapters, hotswapping, distributed PEFT
- **Distributed training** (`references/distributed.md`): multi-GPU DDP, FSDP, DeepSpeed ZeRO-2/3, tensor parallelism, sequence parallelism, debugging
