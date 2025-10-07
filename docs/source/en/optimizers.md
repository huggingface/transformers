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

# Optimizers

Transformers offers two native optimizers, AdamW and AdaFactor. It also provides integrations for more specialized optimizers. Install the library that offers the optimizer and drop it in the `optim` parameter in [`TrainingArguments`].

This guide will show you how to use these optimizers with [`Trainer`] using [`TrainingArguments`] shown below.

```py
import torch
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, Trainer

args = TrainingArguments(
    output_dir="./test-optimizer",
    max_steps=1000,
    per_device_train_batch_size=4,
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-5,
    save_strategy="no",
    run_name="optimizer-name",
)
```

## APOLLO

```bash
pip install apollo-torch
```

[Approximated Gradient Scaling for Memory Efficient LLM Optimization (APOLLO)](https://github.com/zhuhanqing/APOLLO) is a memory-efficient optimizer that allows full parameter learning for both pretraining and fine-tuning. It maintains AdamW-level performance with SGD-like memory efficiency. For extreme memory efficiency, you can use APOLLO-Mini, a rank 1 variant of APOLLO. APOLLO optimizers support:

* Ultra-low rank efficiency. You can use a much lower rank than [GaLoRE](./trainer#galore), rank 1 is sufficient.
* Avoid expensive SVD computations. APOLLO leverages random projections to avoid training stalls.

Use the `optim_target_modules` parameter to specify which layers to train.

```diff
import torch
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./test-apollo",
    max_steps=100,
    per_device_train_batch_size=2,
+   optim="apollo_adamw",
+   optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-5,
    save_strategy="no",
    run_name="apollo_adamw",
)
```

For additional training options, use `optim_args` to define hyperparameters like `rank`, `scale`, and more. Refer to the table below for a complete list of available hyperparameters.

> [!TIP]
> The `scale` parameter can be set to `n/r`, where `n` is the original space dimension and `r` is the low-rank space dimension. You could achieve a similar effect by adjusting the learning rate while keeping `scale` at its default value.

| parameter | description | APOLLO | APOLLO-Mini |
|---|---|---|---|
| rank | rank of the auxiliary sub-space for gradient scaling | 256 | 1 |
| scale_type | how scaling factors are applied | `channel` (per-channel scaling) | `tensor` (per-tensor scaling) |
| scale | adjusts gradient updates to stabilize training | 1.0 | 128 |
| update_proj_gap | steps before updating projection matrices | 200 | 200 |
| proj | projection type | `random` | `random` |

The example below enables the APOLLO-Mini optimizer.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./test-apollo_mini",
    max_steps=100,
    per_device_train_batch_size=2,
    optim="apollo_adamw",
    optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
    optim_args="proj=random,rank=1,scale=128.0,scale_type=tensor,update_proj_gap=200",
)
```

## GrokAdamW

```bash
pip install grokadamw
```

[GrokAdamW](https://github.com/cognitivecomputations/grokadamw) is an optimizer designed to help models that benefit from *grokking*, a term used to describe delayed generalization because of slow-varying gradients. It is particularly useful for models requiring more advanced optimization techniques to achieve better performance and stability.

```diff
import torch
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./test-grokadamw",
    max_steps=1000,
    per_device_train_batch_size=4,
+   optim="grokadamw",
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-5,
    save_strategy="no",
    run_name="grokadamw",
)
```

## LOMO

```bash
pip install lomo-optim
```

[Low-Memory Optimization (LOMO)](https://github.com/OpenLMLab/LOMO) is a family of optimizers, [LOMO](https://huggingface.co/papers/2306.09782) and [AdaLomo](https://hf.co/papers/2310.10195), designed for low-memory full-parameter finetuning of LLMs. Both LOMO optimizers fuse the gradient computation and parameter update in one step to reduce memory usage. AdaLomo builds on top of LOMO by incorporating an adaptive learning rate for each parameter like the Adam optimizer.

> [!TIP]
> It is recommended to use AdaLomo without `grad_norm` for better performance and higher throughput.

```diff
args = TrainingArguments(
    output_dir="./test-lomo",
    max_steps=1000,
    per_device_train_batch_size=4,
+   optim="adalomo",
    gradient_checkpointing=True,
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-6,
    save_strategy="no",
    run_name="adalomo",
)
```

## Schedule Free

```bash
pip install schedulefree
```

[Schedule Free optimizer (SFO)](https://hf.co/papers/2405.15682) replaces the base optimizers momentum with a combination of averaging and interpolation. Unlike a traditional scheduler, SFO completely removes the need to anneal the learning rate.

SFO supports the RAdam (`schedule_free_radam`), AdamW (`schedule_free_adamw`) and SGD (`schedule_free_sgd`) optimizers. The RAdam scheduler doesn't require `warmup_steps` or `warmup_ratio`.

By default, it is recommended to set `lr_scheduler_type="constant"`. Other `lr_scheduler_type` values may also work, but combining SFO optimizers with other learning rate schedules could affect SFOs intended behavior and performance.

```diff
args = TrainingArguments(
    output_dir="./test-schedulefree",
    max_steps=1000,
    per_device_train_batch_size=4,
+   optim="schedule_free_radamw",
+   lr_scheduler_type="constant",
    gradient_checkpointing=True,
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-6,
    save_strategy="no",
    run_name="sfo",
)
```

## StableAdamW

```bash
pip install torch-optimi
```

[StableAdamW](https://huggingface.co/papers/2304.13013) is a hybrid between AdamW and AdaFactor. It ports AdaFactor's update clipping into AdamW, which removes the need for gradient clipping. Otherwise, it behaves as a drop-in replacement for AdamW.

> [!TIP]
> If training on large batch sizes or still observing training loss spikes, consider reducing beta_2 between [0.95, 0.99].

```diff
args = TrainingArguments(
    output_dir="./test-stable-adamw",
    max_steps=1000,
    per_device_train_batch_size=4,
+   optim="stable_adamw",
    gradient_checkpointing=True,
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-6,
    save_strategy="no",
    run_name="stable-adamw",
)
```
