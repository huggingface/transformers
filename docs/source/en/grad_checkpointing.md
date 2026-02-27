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

# Gradient checkpointing

The forward pass typically caches all intermediate activations for the backward pass to reuse. However, activations scale with batch size and sequence length. Gradient checkpointing only saves certain activations and discards the rest. This forces the backward pass to recompute some of the activations on-the-fly as they're needed.

```text
Normal training:
  Forward:   [L1]→[L2]→[L3]→[L4]   (save ALL activations)
  Backward:  ←uses cached activations everywhere

Gradient checkpointing:
  Forward:   [L1]→[L2]→[L3]→[L4]   (save only at checkpoints, discard the rest)
  Backward:  ←reaches L2, recomputes L2→L3 from scratch, uses it, discards it
```

Training will be slower because some activations need to be recomputed, but it reduces activation memory.

Set `gradient_checkpointing=True` to enable. 

```py
from trainer import TrainingArguments

args = TrainingArguments(
    ...,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,    # compose with gradient accumulation if necessary
    gradient_checkpointing=True,
)
```

## Next steps

- Read the [GPU memory usage](./model_memory_anatomy) doc to understand what is driving memory usage on the GPU during training.
