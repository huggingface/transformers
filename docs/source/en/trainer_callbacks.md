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

# Callbacks

[`TrainerCallback`] hooks into specific training events (epoch start, evaluation, training end) to modify training state or control flow. Use callbacks to log metrics to experiment trackers like Trackio, customize when saving and evaluation happen, or add other custom behavior. Stack multiple callbacks to combine features.

Callbacks can't modify the training loop itself, like the forward pass. To change what [`Trainer`] computes, [subclass](./trainer_customize) its methods instead.

The diagram below shows every event a callback can hook into.

```md
on_train_begin
  └─ for each epoch:
      on_epoch_begin
      └─ for each step:
          on_step_begin
          ├─ for each gradient accumulation substep:
          │   on_substep_end
          on_pre_optimizer_step   ← after gradient clipping, before optimizer.step()
          on_optimizer_step       ← after optimizer.step()
          on_step_end
          └─ (conditionally):
              on_log
              on_evaluate
              on_save
      on_epoch_end
on_train_end
on_predict / on_prediction_step
on_push_begin
```

## Creating a callback

Subclass [`TrainerCallback`] and override one or more event methods from the diagram above. The example below demonstrates three hooks:

- `on_epoch_begin` prints the current epoch from [`TrainerState`] (a live value) and the learning rate from [`TrainingArguments`] (a static value).
- `on_step_end` reads the current learning rate from the lr_scheduler (passed via `**kwargs`) and sets `should_evaluate` on [`TrainerControl`] (see [`TrainerControl`] for a complete list of control objects) to trigger an evaluation when it drops below the threshold.
- `on_train_end` prints the best metric and the step where it occurred.

```python
from transformers import TrainerCallback

class EpochLoggerCallback(TrainerCallback):
    def __init__(self, lr_eval_threshold=1e-5):
        self.lr_eval_threshold = lr_eval_threshold

    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"Starting epoch {int(state.epoch) + 1}/{state.num_train_epochs} "
              f"(lr={args.learning_rate})")

    def on_step_end(self, args, state, control, **kwargs):
        lr_scheduler = kwargs.get("lr_scheduler")
        if lr_scheduler is not None:
            current_lr = lr_scheduler.get_last_lr()[0]
            if current_lr < self.lr_eval_threshold:
                control.should_evaluate = True

    def on_train_end(self, args, state, control, **kwargs):
        print(f"Training complete! Best metric: {state.best_metric} at step {state.best_global_step}")
```

Register the callback with [`Trainer`] in the `callbacks` argument. You can also pass multiple callbacks as a list.

```python
trainer = Trainer(
    callbacks=[EpochLoggerCallback(lr_eval_threshold=1e-5)],
    ...,
)
```

## Built-in callbacks

Transformers includes several built-in callbacks that are active by default. Additional [integrated callbacks](./main_classes/callback#available-callbacks) log to platforms like [Trackio](https://huggingface.co/docs/trackio/en/index).

### DefaultFlowCallback

[`DefaultFlowCallback`] manages the default logging, evaluation, and checkpoint schedule based on the `logging_strategy`, `eval_strategy`, and `save_strategy` values in [`TrainingArguments`]. At the right step or epoch, it sets the corresponding `control` flags (`should_log`, `should_evaluate`, `should_save`). It also sets `should_training_stop` when `global_step` reaches `max_steps`.

Overriding this callback is the main way to customize *when* logging, evaluation, or saving happens.

### ProgressCallback and PrinterCallback

[`Trainer`] automatically picks between these two callbacks based on the [disable_tqdm](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.disable_tqdm) field in [`TrainingArguments`].

- [`ProgressCallback`] is used by default. It displays a tqdm progress bar during training and a separate bar during evaluation or prediction, and prints the latest metrics on each `on_log` event. During distributed training, it only runs on the main process to avoid duplicate output.
- [`PrinterCallback`] is used when `disable_tqdm=True`. It prints the log dictionary to stdout on every `on_log` event with no progress bar.

You can also swap them manually with [`~Trainer.remove_callback`] and [`~Trainer.add_callback`].

```py
from transformers import PrinterCallback

trainer = Trainer(...)
trainer.remove_callback(ProgressCallback)
trainer.add_callback(PrinterCallback)
```

### EarlyStoppingCallback

[`EarlyStoppingCallback`] stops training when an evaluation metric stops improving. After each evaluation, it checks whether the metric improved by more than `early_stopping_threshold`. If the metric hasn't improved for `early_stopping_patience` consecutive evaluations, training stops.

[`EarlyStoppingCallback`] requires two [`TrainingArguments`]:

- `metric_for_best_model`, the evaluation metric to monitor
- `eval_strategy`, whether to evaluate on `"steps"` or `"epoch"`

```py
from transformers import EarlyStoppingCallback

trainer = Trainer(
    ...,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],
)
```

## Next steps

- See all available [integrated callbacks](./main_classes/callback#available-callbacks) for logging to experiment trackers.
- The [Subclassing Trainer methods](./trainer_customize) guide covers overriding [`Trainer`] methods when you need to change what the training loop computes.
