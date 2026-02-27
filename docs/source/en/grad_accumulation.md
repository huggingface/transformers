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

Large batches result in large activations that can cause your GPU to run out of memory. Gradient accumulation enables training with a larger effective batch size without storing all the gradients in GPU memory at once.

The gradients are "accumulated" or summed across *n* mini-batches before the optimizer step. For example, if your GPU can only fit a batch of 8 and you accumulate for 4 steps, the effective batch size is 32.

```text
Step 1: mini-batch 1 → forward → backward → grads = G₁
Step 2: mini-batch 2 → forward → backward → grads = G₁ + G₂
Step 3: mini-batch 3 → forward → backward → grads = G₁ + G₂ + G₃
Step 4: mini-batch 4 → forward → backward → grads = G₁ + G₂ + G₃ + G₄
        → optimizer.step()  ← same update as if batch_size × 4
        → zero_grad()
```

Throughput is lower with gradient accumulation because the optimizer runs less frequently, so you should only use it if memory is an issue.

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

[`Trainer`] automatically divides loss by `gradient_accumulation_steps` before the backward pass. But if you're using a [custom loss function](./trainer_recipes#custom-loss-function) that accepts `num_items_in_batch` (total non-padding token count across all mini-batches), you need to handle the loss normalization yourself.

```py
import torch
import torch.nn.functional as F

def my_loss_fn(outputs, labels, num_items_in_batch=None):
    logits = outputs["logits"]
    if num_items_in_batch is not None:
        loss = F.cross_entropy(logits, labels, reduction="sum")
        loss = loss / num_items_in_batch                           # scale loss
    else:
        loss = F.cross_entropy(logits, labels, reduction="mean")
    return loss
```

## Next steps

- Read the [GPU memory usage](./model_memory_anatomy) doc to understand what is driving memory usage on the GPU during training.
- See the [Gradient checkpointing](./grad_checkpointing) guide to learn how to reduce activation memory by recomputing activations instead of caching them.
- See the [Mixed precision training](./mixed_precision_training) guide to learn how to use lower precision data types to reduce memory and speed up training.
