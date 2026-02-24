<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Subclassing Trainer methods

Subclass [`Trainer`] methods to change training behavior without rewriting the entire loop. Subclassing modifies the *training loop*, for example the forward pass or loss computation.

Before subclassing, consider whether you need to change *what* [`Trainer`] computes or *when* and *whether* it acts. For timing and conditional logic, use a [Callback](./trainer_callbacks) instead. Callbacks control when things happen (logging, evaluation, early stopping) and subclassing changes what happens (loss computation, data loading, optimization).

> [!NOTE]
> See the [`Trainer`] API docs for a complete list of methods you can subclass. Private methods (prefixed with `_`) like `_save_checkpoint` or `_evaluate` can also be overridden, but these may change without notice.

## get_train_dataloader

The standard [`~Trainer.get_train_dataloader`] method loads one batch, trains on it, discards it, and loads the next batch.

```py
def get_train_dataloader(self):
    return self._get_dataloader(
        batch_size=self._train_batch_size,
        ...
)
```

[GRPO](https://huggingface.co/docs/trl/en/grpo_trainer) is an online reinforcement learning algorithm that generates completions before training on them. Generating completions every step is expensive because it's autoregressive. A 512-token completion requires ~512 sequential forward passes compared to one forward pass for a training step. [`~trl.GRPOTrainer`] subclasses [`~Trainer.get_train_dataloader`] to batch generation across multiple steps.

[`trl.GRPOTrainer.get_train_dataloader`] loads *batches* of generation prompts for multiple training steps at once by multiplying batch size by a `steps_per_generation` argument. If `train_batch_size=4` and `steps_per_generation=8`, the dataloader produces batches of 32, cutting generation cost by 8x.

```py
def get_train_dataloader(self):
    dataloader_params = {
        "batch_size": self._train_batch_size * self.args.steps_per_generation, # this is the only change
        ...
    }
```

## compute_loss

[`~Trainer.compute_loss`] returns the cross-entropy loss calculated by the model.

```py
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    ...
    outputs = model(**inputs)
    ...
    loss = outputs["loss"] # get loss from model

    return (loss, outputs) if return_outputs else loss
```

[DPO](https://huggingface.co/docs/trl/en/dpo_trainer) measures how strongly the policy model prefers a chosen response over a rejected one, relative to a reference model. [`~trl.DPOTrainer`] subclasses [`~Trainer.compute_loss`] because the loss computation differs from standard cross-entropy in several ways:

- the model never sees labels; it only returns logits for DPO to calculate log-probs from
- chosen and rejected responses are concatenated
- a reference model calculates its own log-probs
- the loss is a function of `π_chosen`, `π_rejected`, `π_ref_chosen`, `π_ref_rejected`

None of the above fits the standard [`Trainer.compute_loss`] method.

```py
def compute_loss(
    self,
    model: PreTrainedModel | nn.Module,
    inputs: dict[str, torch.Tensor | Any],
    return_outputs=False,
    num_items_in_batch=None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    ...
    outputs = model(**inputs)
    logits = outputs.logits
    logps = get_logps(logits, inputs)
    chosen_logps, rejected_logps = logps.chunk(2, dim=0)  # batch is [chosen, rejected]
    ref_logits = self.ref_model(**inputs).logits
    ref_logps = get_logps(ref_logits, inputs)
    ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)  # batch is [chosen, rejected]
    chosen_scores = chosen_logps - ref_chosen_logps
    rejected_scores = rejected_logps - ref_rejected_logps
    per_sequence_loss = -F.logsigmoid(self.beta * chosen_scores - rejected_scores)
    loss = per_sequence_loss.mean()
    return (loss, outputs) if return_outputs else loss
```

## Next steps

- For more real-world examples, see how [`~trl.GRPOTrainer`] and [`~trl.DPOTrainer`] extend [`Trainer`] in TRL, or how [Axolotl](https://github.com/axolotl-ai-cloud/axolotl/tree/main/src/axolotl/core/trainers) builds custom trainers on top of it.
