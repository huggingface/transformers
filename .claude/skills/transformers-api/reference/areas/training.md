# Training / Fine-tuning (Trainer + Seq2SeqTrainer)

## Contents
- [Scope](#scope)
- [Minimum questions to ask](#minimum-questions-to-ask)
- [Decision guide: `Trainer` vs `Seq2SeqTrainer` vs custom loop](#decision-guide-trainer-vs-seq2seqtrainer-vs-custom-loop)
- [Quickstarts](#quickstarts)
  - [1) Trainer: text classification (baseline + eval)](#1-trainer-text-classification-baseline--eval)
  - [2) Trainer: map/tokenize a Dataset safely (columns + labels)](#2-trainer-maptokenize-a-dataset-safely-columns--labels)
  - [3) Trainer: distributed / multi-GPU launch (Accelerate/torchrun)](#3-trainer-distributed--multi-gpu-launch-acceleratetorchrun)
  - [4) Trainer: image classification (non-text example; `remove_unused_columns=False`)](#4-trainer-image-classification-non-text-example-remove_unused_columnsfalse)
  - [5) Trainer: custom loss (minimal override)](#5-trainer-custom-loss-minimal-override)
  - [6) Trainer: evaluate/predict-only (no training)](#6-trainer-evaluatepredict-only-no-training)
- [Knobs that matter (3–8)](#knobs-that-matter-38)
- [Pitfalls & fixes](#pitfalls--fixes)
- [Column dropping and why it matters](#column-dropping-and-why-it-matters)
- [Verify / locate in repo](#verify--locate-in-repo)

---

## Scope

Use this page when the user wants to **fine-tune / train / evaluate** a model in `transformers` using `Trainer` or `Seq2SeqTrainer`.

---

## Minimum questions to ask

Ask only what you need to produce a runnable snippet (0–6 questions):
1) **Task** (classification / token classification / seq2seq / causal LM / vision / audio)
2) **Model id or local path** (and `revision` if pinned)
3) **Dataset** source + columns (inputs, labels, any extra metadata needed)
4) **Backend + device** (PyTorch; CPU/CUDA/MPS; num GPUs; rough VRAM)
5) **Goal** (correctness vs speed vs memory vs reproducibility)
6) If blocked: **full traceback + exact versions** + smallest repro

---
## Decision guide: `Trainer` vs `Seq2SeqTrainer` vs custom loop

### Prefer `Trainer` when…
- You want the **standard, feature-complete** training/eval loop with minimal custom code. 
- Your evaluation can be done from a **forward pass** (loss/logits → `compute_metrics`), optionally with `preprocess_logits_for_metrics` to transform logits before metrics caching.
- You may still be doing seq2seq *training*, but you **don’t need `generate()` during eval/predict** (e.g., loss-based evaluation only). 

### Prefer `Seq2SeqTrainer` when…
- You’re training **sequence-to-sequence** models (e.g., summarization/translation) and want the seq2seq-adapted training path.
- You want evaluation/prediction **with generation** (`predict_with_generate=True`) so you can compute ROUGE/BLEU-style metrics from generated sequences. 
- You want easy control over generation at eval/predict time (e.g., `max_length`, `num_beams`, and other `generate` kwargs). 

### Prefer a custom loop when…
- You need **nonstandard optimizer steps**, RL-style objectives, multi-stage losses, or very custom batching/updates that don’t fit cleanly into Trainer customization.
- You’re ready to write your own loop (often with **Accelerate** to avoid distributed/mixed-precision boilerplate). 
---

## Quickstarts

### 1. Trainer: text classification (baseline + eval)

```python
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

model_id = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

ds = load_dataset("imdb")
tok = AutoTokenizer.from_pretrained(model_id)

def preprocess(batch):
    return tok(batch["text"], truncation=True)

tok_ds = ds.map(preprocess, batched=True, remove_columns=["text"])

if "label" in tok_ds["train"].column_names and "labels" not in tok_ds["train"].column_names:
    tok_ds = tok_ds.rename_column("label", "labels")

train_ds = tok_ds["train"].shuffle(seed=42).select(range(2000))
eval_ds  = tok_ds["test"].shuffle(seed=42).select(range(2000))

model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
collator = DataCollatorWithPadding(tokenizer=tok)

def compute_metrics(eval_pred):
    logits = eval_pred.predictions if hasattr(eval_pred, "predictions") else eval_pred[0]
    labels = eval_pred.label_ids if hasattr(eval_pred, "label_ids") else eval_pred[1]
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}

args = TrainingArguments(
    output_dir="./out_cls",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="no",
    save_strategy="no",       
    load_best_model_at_end=False,
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tok,      
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()
print(trainer.evaluate())
trainer.save_model("./out_cls/final")
```

Notes:
- If you don’t want eval, set `eval_strategy="no"` and omit `eval_dataset`.
- Start by training on a small sample (e.g., 200–2,000 examples) to quickly verify the pipeline runs end-to-end before scaling to the full dataset.
---

### 2. Trainer: map/tokenize a Dataset safely (columns + labels)

This checklist prevents 80% of “why is loss None / labels missing / shapes wrong” issues.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

model_id = "distilbert/distilbert-base-uncased"
ds = load_dataset("imdb")
tok = AutoTokenizer.from_pretrained(model_id)

def preprocess(batch):
    out = tok(batch["text"], truncation=True)
    out["labels"] = batch["label"]     # make supervision explicit
    return out

proc = ds["train"].map(preprocess, batched=True, remove_columns=["text"])

ex = proc[0]
print(sorted(ex.keys()))
print("len(input_ids):", len(ex["input_ids"]), "labels:", ex["labels"])
```

If you have multiple supervision fields (e.g., `start_positions`/`end_positions` or multi-task),
keep them as explicit columns and handle them via your model forward and/or `label_names` (advanced).

---

### 3. Trainer: distributed / multi-GPU launch (Accelerate/torchrun)

Trainer typically scales via the launcher you use (code often stays the same).

**Option A: Accelerate**
```bash
accelerate config
accelerate launch train.py
```

**Option B: torchrun**
```bash
torchrun --nproc_per_node 2 train.py
```

Practical scaling knobs:
- Reduce per-device batch size and use `gradient_accumulation_steps` to keep the same global batch.
- For instability, start with fewer GPUs and confirm correctness first.

---

### 4. Trainer: image classification (non-text example; `remove_unused_columns=False`)

For vision/video, you often need the raw `image`/`video` column to build `pixel_values`.
Trainer may drop columns by default, so set `remove_unused_columns=False`.

```python
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    TrainingArguments,
    Trainer,
)

model_id = "google/vit-base-patch16-224"
ds = load_dataset("beans")  # has an `image` column

processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id, num_labels=3)

def transform(example):
    # example["image"] is a PIL image
    enc = processor(example["image"], return_tensors="pt")
    example["pixel_values"] = enc["pixel_values"][0]
    if "label" in example and "labels" not in example:
        example["labels"] = example["label"]
    return example

train_ds = ds["train"].with_transform(transform)
eval_ds  = ds["validation"].with_transform(transform)

args = TrainingArguments(
    output_dir="./out_vit",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    learning_rate=5e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    remove_unused_columns=False,   # IMPORTANT for transforms that rely on raw columns
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=processor,
    data_collator=DefaultDataCollator(),
)

trainer.train()
print(trainer.evaluate())
```

---

### 5. Trainer: custom loss (minimal override)

Use this when you need a custom loss but want to keep Trainer’s loop.

```python
import torch
from transformers import Trainer

class CustomLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Example: multi-label BCE loss (labels should be float multi-hot)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

        return (loss, outputs) if return_outputs else loss
```

Then use it like `Trainer`:
```python
# trainer = CustomLossTrainer(model=..., args=..., train_dataset=..., eval_dataset=..., ...)
```

---

### 6. Trainer: evaluate/predict-only (no training)

Useful for smoke tests, regression checks, or “just compute metrics”.

```python
# assume you already built: trainer = Trainer(...)
metrics = trainer.evaluate()
print("eval:", metrics)

pred = trainer.predict(trainer.eval_dataset)
print("metrics:", pred.metrics)
print("predictions shape:", getattr(pred.predictions, "shape", None))
```

---

## Knobs that matter (3–8)

Prioritize these knobs before anything else:

1) **Task ↔ model head compatibility**
   - classification → `AutoModelForSequenceClassification`
   - seq2seq → `AutoModelForSeq2SeqLM` + `Seq2SeqTrainer`
2) **`model` + `revision`** (pin for reproducibility)
3) **Data correctness**
   - label key: prefer `labels`
   - correct dtypes/shapes (class ids vs multi-hot vs token ids)
4) **Batching vs memory**
   - `per_device_train_batch_size`, `gradient_accumulation_steps`
5) **Evaluation/save cadence**
   - `eval_strategy`, `eval_steps`, `save_strategy`, `save_steps`
6) **Precision**
   - `fp16` / `bf16` (if supported)
7) **Column handling**
   - `remove_unused_columns` (often needs `False` for vision/video or custom transforms)
8) **Best model selection**
   - `load_best_model_at_end`, `metric_for_best_model`, `greater_is_better`

---

## Pitfalls & fixes

- **TypeError: unexpected keyword**
  - `eval_strategy` → try `evaluation_strategy`
  - `processing_class` → try `tokenizer`
- **Eval enabled but no eval dataset**
  - Provide `eval_dataset`, or set `eval_strategy="no"`.
- **Loss is `None` / labels ignored**
  - Ensure the label key is `labels` and its dtype matches the loss (int class ids vs float multi-hot).
- **Trainer drops columns you still need**
  - Set `remove_unused_columns=False` and manage inputs carefully (especially vision/video transforms).
- **OOM**
  - Reduce batch size, increase `gradient_accumulation_steps`, lower precision, shorten sequence lengths.
  - For deep tuning route to `reference/areas/performance.md`.
- **Very slow “time to first step”**
  - Dataset transforms/caching/dataloader workers can dominate; start with a tiny subset and `num_workers=0`.

---

## Column dropping and why it matters

By default, Trainer removes dataset columns that aren’t accepted by `model.forward()`.

This is usually helpful, but it can break workflows where:
- you need raw columns to build model inputs (e.g., `image` → `pixel_values`)
- you keep metadata columns for metrics/debugging

What to do:
- If your preprocessing happens in a dataset transform (e.g., `with_transform`) and needs raw columns:
  - set `TrainingArguments(remove_unused_columns=False)`
- Ensure your transform or collator produces exactly the tensors the model expects.

---

## Verify / locate in repo

Common repo hotspots:
- Trainer loop + internals:
  - `src/transformers/trainer.py`
  - `src/transformers/trainer_utils.py`
  - `src/transformers/trainer_callback.py`
- Seq2Seq training:
  - `src/transformers/trainer_seq2seq.py`
  - `src/transformers/training_args_seq2seq.py`
- Training args + defaults:
  - `src/transformers/training_args.py`
- Collators:
  - `src/transformers/data/data_collator.py`
- Integrations (DeepSpeed/FSDP/etc.):
  - `src/transformers/integrations/`