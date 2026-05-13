<!---Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Gradient accumulation

Large batches produce large activations that exhaust GPU memory. Gradient accumulation lets you train with a larger effective batch size by spreading gradient computation across multiple mini-batches.

Gradients accumulate across *n* mini-batches before the optimizer updates the weights. For example, with a per-device batch size of 8 and 4 accumulation steps, the effective batch size is 32.

```text
Step 1: mini-batch 1 → forward → backward → grads = G₁
Step 2: mini-batch 2 → forward → backward → grads = G₁ + G₂
Step 3: mini-batch 3 → forward → backward → grads = G₁ + G₂ + G₃
Step 4: mini-batch 4 → forward → backward → grads = G₁ + G₂ + G₃ + G₄
        → optimizer.step()  ← same update as if batch_size × 4
        → zero_grad()
```

Use gradient accumulation only when a larger batch doesn't fit in memory. It doesn't improve throughput over training with a true large batch.

Accumulate gradients for `gradient_accumulation_steps` across `per_device_train_batch_size`.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    ...,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
)
```

## Loss scaling

For a [custom loss function](./trainer_recipes#custom-loss-function), include `num_items_in_batch` so [`Trainer`] divides the loss by the total non-padding token count across all mini-batches. This normalizes by tokens rather than a fixed step count with `gradient_accumulation_steps`. Otherwise, [`Trainer`] divides loss by `gradient_accumulation_steps`.

```py
import torch.nn.functional as F

def compute_loss(outputs, labels, num_items_in_batch=None):
    logits = outputs["logits"]
    loss = F.cross_entropy(logits, labels, reduction="sum")
    return loss / num_items_in_batch
```

## Next steps

- Read the [GPU memory usage](./model_memory_anatomy) doc to understand what is driving memory usage on the GPU during training.
- See the [Gradient checkpointing](./grad_checkpointing) guide to learn how to reduce activation memory by recomputing activations instead of caching them.
- See the [Mixed precision training](./mixed_precision_training) guide to learn how to use lower precision data types to reduce memory and speed up training.
- Read the [Gradient Accumulation Fix](https://unsloth.ai/blog/gradient) blog post to learn how gradient accumulation is computed.
