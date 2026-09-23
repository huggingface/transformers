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

Training is typically slower because the backward pass recomputes discarded activations, but checkpointing reduces
activation memory.

Set `gradient_checkpointing=True` to enable.

> [!TIP]
> Use with [gradient accumulation](./grad_accumulation) to further reduce memory usage.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    ...,
    gradient_checkpointing=True,
)
```

## Partial checkpointing

Full gradient checkpointing recomputes every checkpointable layer. If your run has some memory headroom, checkpoint
fewer layers to trade some of the memory savings for speed.

Pass `every_n_layers` to [`~PreTrainedModel.gradient_checkpointing_enable`] to choose the
checkpointing interval.

```text
every_n_layers=2

Forward:   input  -> [L1] -> [L2] -> [L3] -> [L4] -> [L5] -> [L6]
                      CP     keep     CP     keep     CP     keep

Backward:  output <- [L6] <- [L5] <- [L4] <- [L3] <- [L2] <- [L1]
                     keep   rerun    keep   rerun    keep   rerun
```

With `every_n_layers=2`, the first layer and every second layer after it are checkpointed. Checkpointed layers discard
their activations during the forward pass and recompute them during the backward pass, while the other layers keep their
activations in memory.

```py
model.gradient_checkpointing_enable(every_n_layers=2)
```

The default, `every_n_layers=1`, checkpoints every layer. Larger values checkpoint the first layer and then every `n`
layers after it, leaving the other layers' activations in memory. For example, `every_n_layers=2` checkpoints
layers 1, 3, 5, and so on. Only modules that inherit from [`GradientCheckpointingLayer`] are counted. Other modules
that support gradient checkpointing remain enabled.

To use partial gradient checkpointing with [`Trainer`], set `every_n_layers` in `gradient_checkpointing_kwargs`.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    ...,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"every_n_layers": 4},
)
```

<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/partial-grad-ckpt.png" />
</div>

## Offloading the saved activations

Gradient checkpointing keeps one activation per checkpointed layer on the GPU. At long sequence lengths, this can
consume substantial memory (approximately `layers x sequence x hidden x bytes_per_element`). Set `offload` to hold
those activations in pinned host memory instead, at the cost of a device-to-host copy during the forward pass and a
host-to-device copy during the backward pass.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    ...,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"offload": True},
)
```

Both copies run on the compute stream, so this trades a slower step for the memory. Reach for it when a run does not fit otherwise, not to speed one up.

## Next steps

- Read the [GPU memory usage](./model_memory_anatomy) doc to understand what is driving memory usage on the GPU during training.
- See the [Mixed precision training](./mixed_precision_training) guide to learn how to use lower precision data types to further reduce memory and speed up training.
- See the [Kernels](./kernels) guide to learn how to speed up training with custom fused kernels.
