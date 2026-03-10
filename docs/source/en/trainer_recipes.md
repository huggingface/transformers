<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Trainer features

Each recipe below demonstrates a specific [`Trainer`] feature: custom loss functions, memory-efficient evaluation, checkpointing strategies, and more.

> [!TIP]
> Open an [issue](https://github.com/huggingface/transformers/issues/new/choose) if there is a feature or workflow you'd like to see here.

## Custom loss function

Pass [compute_loss_func](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.Trainer.compute_loss_func) to [`Trainer`] to replace the default loss function. The function runs *after* the forward pass and only defines how loss is computed from the outputs. To modify the forward pass itself, [subclass](./trainer_customize#compute_loss) [`~Trainer.compute_loss`] instead.

The custom loss function must have the following signature:

```py
import torch.nn.functional as F

def my_loss_fn(outputs, labels, num_items_in_batch):
    logits = outputs["logits"]
    loss = F.cross_entropy(logits, labels, reduction="sum")
    return loss / num_items_in_batch
```

- `outputs` is the raw model output (`outputs.logits` has shape `(batch, seq_len, vocab_size)`).
- `labels` is the token ids popped from the input batch by [`Trainer`] before the forward pass.
- `num_items_in_batch` is the total non-padding token count across the full accumulated batch. [`Trainer`] skips automatic loss normalization when a custom loss function is provided, so your function must handle normalization directly.

```py
trainer = Trainer(
    model=model,
    args=TrainingArguments(...),
    train_dataset=train_dataset,
    compute_loss_func=my_loss_fn,
)
trainer.train()
```

> [!NOTE]
> See the [subclassing guide](./trainer_customize#compute_loss) for more examples of overriding [`~Trainer.compute_loss`].

## Evaluating on start

Set `eval_on_start=True` to run a full eval pass before the first training step. A pre-training eval surfaces issues with the evaluation pipeline early, especially during long runs.

`eval_on_start` requires a valid `eval_strategy` (such as `"epoch"`) and an eval dataset.

```py
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="out",
        eval_strategy="epoch",
        eval_on_start=True,
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
```

A full eval adds time, so it's most useful on first runs or after modifying `compute_metrics`.

## Memory-efficient evals

During evaluation, [`Trainer`] runs a forward pass on every batch and concatenates the logits into a single tensor on the GPU. Once the eval dataset is fully processed, [`Trainer`] moves the concatenated logits to the CPU and calls `compute_metrics`. For large models or eval sets, the accumulated logits can exhaust GPU memory even when training on the same hardware works fine, because training only holds one batch of activations at a time.

### eval_accumulation_steps

Offload the accumulated predictions from GPU to CPU every *n* batches. Lower values reduce GPU memory at the cost of more frequent CPU transfers.

```py
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="out",
        eval_strategy="epoch",
        eval_accumulation_steps=16,   # move predictions to CPU every 16 batches
    ),
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
```

### preprocess_logits_for_metrics

Called once per eval batch on the GPU, immediately after the forward pass and before logit accumulation. The returned value replaces the logits in `eval_pred.predictions`. Running the computation at the batch level reduces per-batch tensor size and gives `eval_accumulation_steps` a smaller tensor to offload.

```py
import evaluate
from transformers import Trainer, TrainingArguments

metric = evaluate.load("accuracy")

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="out",
        eval_strategy="epoch",
        eval_accumulation_steps=16,
    ),
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)
trainer.train()
```

## Dataloader performance

By default, [`Trainer`] creates a dataloader with `dataloader_num_workers=0`. Data is loaded in the main process while the GPU idles, which shows up as low GPU utilization between batches.

Both `dataloader_persistent_workers` and `dataloader_prefetch_factor` require `dataloader_num_workers > 0`.

- `dataloader_persistent_workers` keeps worker subprocesses alive between epochs to avoid reinitializing from scratch, at the cost of higher memory.
- `dataloader_prefetch_factor` controls how many batches each worker prepares in advance. With `dataloader_prefetch_factor=2` and `num_workers=4`, up to 8 batches sit in memory while the GPU trains on the current one.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="out",
    dataloader_num_workers=4,            # spawn 4 worker subprocesses
    dataloader_persistent_workers=True,  # keep them alive between epochs
    dataloader_prefetch_factor=2,        # each worker preloads 2 batches ahead
)
```

## NEFTune

[NEFTune](https://hf.co/papers/2310.05914) adds random noise to token embeddings during the forward pass. The noise acts as regularization and can improve performance for instruction fine-tuning.

Enable NEFTune by setting `neftune_noise_alpha` in [`TrainingArguments`]. Typical alpha values range from 5 to 15.

```py
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="out",
        num_train_epochs=3,
        neftune_noise_alpha=5,
    ),
    train_dataset=train_dataset,
)
trainer.train()
```

NEFTune only affects training, and the original embedding layer is restored after training.

## Logging

Control when and where [`Trainer`] writes log entries with `logging_strategy`, `logging_steps`, and `report_to`.

- `logging_strategy="steps"` logs every [`~TrainingArguments.logging_steps`] optimizer updates. Use `"epoch"` to log at each epoch end instead.
- `report_to` streams logs to an experiment tracker like Trackio.

```py
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="out",
        logging_strategy="steps",
        logging_steps=50,               # write a log entry every 50 optimizer updates
        report_to="trackio",            # stream to Trackio (or "wandb", "tensorboard", …)
        run_name="model-experiment-v1", # display name in the tracker
    ),
    train_dataset=train_dataset,
)
trainer.train()
```

## Checkpointing

[`Trainer`] saves a checkpoint every [`~TrainingArguments.save_steps`] optimizer update and keeps all of them (or the most recent [`~TrainingArguments.save_total_limit`]).

`save_strategy="best"` keeps only the single best checkpoint according to a metric. A new checkpoint is saved only when the tracked metric improves, which saves disk space and avoids accumulating stale checkpoints.

```py
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="out",
        eval_strategy="epoch",
        save_strategy="best",
        metric_for_best_model="perplexity",   # save when eval perplexity improves
        greater_is_better=False,              # lower perplexity is better
        load_best_model_at_end=True,          # load the best weights after training finishes
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,          # must return {"perplexity": ...}
)
trainer.train()
```

### Resume training

Pass `resume_from_checkpoint=True` to [`~Trainer.train`] if training was interrupted and you'd like to resume without losing progress. Training will resume from the latest checkpoint in `output_dir`.

```py
trainer.train(resume_from_checkpoint=True)
```

Specify a checkpoint path to resume from a particular point.

```py
trainer.train(resume_from_checkpoint="out/checkpoint-1000")
```

When resuming, [`Trainer`] restores the optimizer state, scheduler state, and RNG state.

Checkpoint resuming requires optimizer and scheduler state files in the checkpoint directory. If those files are missing (for example, when `save_only_model=True`), the optimizer restarts from scratch.
