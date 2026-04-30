# Customization reference

This doc outlines customization options for fine-tuning.

## Decision guide

| Need | Mechanism |
|---|---|
| Log metrics, early stop, adjust schedule | `TrainerCallback` |
| Custom loss function (simple) | `compute_loss_func` parameter |
| Change forward pass, dataloader, eval | Subclass `Trainer` |
| Custom batch structure / extra fields | Subclass `DataCollatorWithPadding` or `DataCollatorMixin` |
| Custom optimizer or LR schedule | `optimizer_cls_and_kwargs` or `optimizers=` or subclass `create_optimizer` |
| Search hyperparameters | `Trainer.hyperparameter_search` |

---

## Data collators

### Extend DataCollatorWithPadding (extra fields in batch)

Pop custom fields before calling `super().__call__()` — the parent doesn't know about them:

```python
from dataclasses import dataclass
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

@dataclass
class DataCollatorWithScore(DataCollatorWithPadding):
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features):
        scores = [f.pop("score") for f in features]  # remove before parent processes
        batch = super().__call__(features)
        batch["score"] = torch.tensor(scores, dtype=torch.float)
        return batch
```

### Full control with DataCollatorMixin

Use when batch structure is fundamentally different (e.g., preference pairs, multi-sequence inputs):

```python
from transformers.data.data_collator import DataCollatorMixin

class DataCollatorForPreference(DataCollatorMixin):
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict]) -> dict:
        chosen = [torch.tensor(ex["chosen_ids"]) for ex in examples]
        rejected = [torch.tensor(ex["rejected_ids"]) for ex in examples]
        all_ids = chosen + rejected
        return {
            "input_ids": pad_sequence(all_ids, batch_first=True, padding_value=self.pad_token_id),
            "attention_mask": pad_sequence(
                [torch.ones_like(ids) for ids in all_ids], batch_first=True, padding_value=0
            ),
        }
```

---

## Custom loss

### Without subclassing (preferred for simple cases)

```python
import torch.nn.functional as F

def my_loss(outputs, labels, num_items_in_batch):
    logits = outputs["logits"]
    # Normalize by non-padding token count, not batch size
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="sum")
    return loss / num_items_in_batch

trainer = Trainer(..., compute_loss_func=my_loss)
```

### Via subclassing

```python
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")   # Trainer pops labels before forward — do it here too
        outputs = model(**inputs)
        loss = my_custom_loss(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
```

---

## Callbacks

Callbacks observe and signal — they cannot modify the training loop. Use subclassing for that.

```python
from transformers import TrainerCallback

class MyCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs): ...
    def on_epoch_begin(self, args, state, control, **kwargs): ...
    def on_step_end(self, args, state, control, **kwargs): ...
    def on_evaluate(self, args, state, control, metrics, **kwargs): ...
    def on_save(self, args, state, control, **kwargs): ...
    def on_log(self, args, state, control, logs, **kwargs): ...
    def on_train_end(self, args, state, control, **kwargs): ...
```

Key objects in all hooks:
- `args` — `TrainingArguments` (static config)
- `state` — live values: `global_step`, `epoch`, `best_metric`, `log_history`
- `control` — set flags: `should_training_stop`, `should_evaluate`, `should_save`, `should_log`

Built-in callbacks:
- `EarlyStoppingCallback(early_stopping_patience=3)` — requires `metric_for_best_model` **and** `load_best_model_at_end=True` in TrainingArguments
- `DefaultFlowCallback` — controls when logging/eval/save happen; override to change cadence

```python
trainer = Trainer(..., callbacks=[MyCallback()])
```

---

## Subclassing Trainer

For changing computation (loss, forward pass, dataloader), not just observing events.

```python
class MyTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=..., sampler=MySampler(...))
```

Avoid overriding private methods (`_save_checkpoint`, `_evaluate`) — they can change without notice.

---

## Optimizers and schedulers

### Built-in optimizer choices (`optim=`)

Common values: `"adamw_torch"` (default), `"adamw_torch_fused"`, `"adamw_8bit"` (bitsandbytes), `"adafactor"`, `"sgd"`, `"apollo_adamw"`, `"adalomo"`, `"schedule_free_radam"`

```python
TrainingArguments(
    optim="adamw_8bit",
    learning_rate=2e-5,
    weight_decay=0.01,
    adam_beta1=0.9, adam_beta2=0.999,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    lr_scheduler_kwargs={},   # scheduler-specific extra params
)
```

### Pass optimizer class + kwargs (Trainer builds it after model is on device)

```python
trainer = Trainer(
    ...,
    optimizer_cls_and_kwargs=(torch.optim.SGD, {"momentum": 0.9, "nesterov": True}),
)
```

### Pass prebuilt optimizer + scheduler

**Build AFTER the model is on its device** — parameters resolve at construction time and device mismatches cause silent failures in distributed training:

```python
model = model.to(device)   # must be on device first
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=10_000)
trainer = Trainer(..., optimizers=(optimizer, scheduler))
```

### Metric-based LR scheduler (GreedyLR)

Reduces LR when a metric stops improving — useful when you don't know the ideal schedule in advance:

```python
TrainingArguments(
    lr_scheduler_type="greedy",
    lr_scheduler_kwargs={"patience": 10, "factor": 0.95, "min_lr": 1e-5},
    eval_strategy="steps",
    eval_steps=200,
)
```

### Per-layer learning rates via subclass

```python
class MyTrainer(Trainer):
    def create_optimizer(self):
        super().create_optimizer()
        self.optimizer.add_param_group({
            "params": self.model.classifier.parameters(),
            "lr": self.args.learning_rate * 10,
        })
        return self.optimizer
```

---

## Hyperparameter search

Use `model_init` (not `model=`) so a fresh model is created for each trial:

```python
def model_init(trial):
    return AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

trainer = Trainer(model_init=model_init, args=args, ...)
```

Install a backend (`pip install optuna` / `ray[tune]` / `wandb`) then define a search space:

```python
def hp_space(trial):   # Optuna example
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
    }

best_run = trainer.hyperparameter_search(
    hp_space=hp_space,
    compute_objective=lambda metrics: metrics["eval_loss"],
    n_trials=20,
    direction="minimize",
    backend="optuna",
)

print(best_run.hyperparameters)
```

Gotcha: `model_init` and `model=` are mutually exclusive — passing both raises an error.
