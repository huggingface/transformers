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

# Padding-free training

Padding-free training (also called packing) concatenates several samples into a single sequence instead of padding each one to a fixed length. The model needs to know where each sample ends so attention doesn't mix tokens across samples.

There are two ways to provide those boundaries.

- Infer them from `position_ids` at runtime.
- Prepare them ahead of time with a data collator.

The recommended approach is the data collator. This guide explains why and covers the caveats of the `position_ids` path.

> [!TIP]
> Padding-free relies on a FlashAttention implementation. Install [flash-attn](https://github.com/Dao-AILab/flash-attention) and load the model with `attn_implementation="flash_attention_2"` (or `"flash_attention_3"`), since only the FlashAttention kernels expose the variable-length path that a flattened batch needs.

## Infer boundaries from position_ids

FlashAttention can detect padding-free batches from `position_ids` alone and remains for backward compatibility, because downstream frameworks such as TRL depend on it.

Relying on `position_ids` has two problems.

- Detecting packed sequences from `position_ids` is a dynamic, data-dependent check. It works without compilation, but under `torch.compile` it causes graph breaks. The check is currently restricted to `batch_size == 1` to limit how often it runs, since real batch sizes are usually larger.
- Compiled FlashAttention forces some kwargs to be plain Python `int`s. Inferring them from `position_ids` at runtime forces device-to-host syncs, and on older PyTorch versions an extra graph break from the tensor-to-int conversion.

## Prepare boundaries with a data collator

Preparing the boundary kwargs up front removes the problems above and behaves identically whether or not you compile.

Use [`DataCollatorWithFlattening`] to flatten each batch and return the boundary information. Set `return_flash_attn_kwargs=True` so the collator precomputes the boundaries instead of leaving them to be inferred from `position_ids` at runtime. Pass it to [`Trainer`] and don't add an `attention_mask`, since the flattened batch already encodes the boundaries and a mask conflicts with the packed layout.

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithFlattening, Trainer, TrainingArguments

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
    device_map="cuda",
)

dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
dataset = dataset.map(
    lambda example: tokenizer(example["text"], truncation=True, max_length=512),
    remove_columns=dataset.column_names,
)

# return_flash_attn_kwargs=True precomputes the sequence boundaries
data_collator = DataCollatorWithFlattening(return_flash_attn_kwargs=True)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="padding-free-llama"),
    train_dataset=dataset,
    data_collator=data_collator,
)
trainer.train()
```

## Linear attention and convolution models

Gated DeltaNet (GDN), other linear-attention layers, and causal convolutions have no `position_ids`-only path, by design. Preparing the data with the collator is the only supported option for these models.

> [!WARNING]
> Don't rely on `position_ids` alone for GDN, linear-attention, or causal convolution models. Prepare the boundary kwargs, including `seq_idx`, with the data collator.

For these models, set both `return_flash_attn_kwargs=True` and `return_seq_idx=True`.

```python
from transformers import DataCollatorWithFlattening

data_collator = DataCollatorWithFlattening(
    return_flash_attn_kwargs=True,
    return_seq_idx=True,
)
```

These models also need their kernels installed, [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) for the linear-attention recurrence and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) for the convolution. Without them, the model falls back to reference implementations that ignore the boundary kwargs and mix tokens across samples.

When the boundary kwargs are missing, the kernels quietly treat the whole batch as one sequence. Nothing raises an error or warning, because a runtime check would add a data-dependent branch that conflicts with `torch.compile`.

## Next steps

- See the [data collators](./data_collators) guide for other collators.
- Browse the [`DataCollatorWithFlattening`] API reference for the full set of arguments.
- Read [Improving Hugging Face Training Efficiency Through Packing with Flash Attention](https://huggingface.co/blog/packing-with-FA2) for benchmarks and a deeper walkthrough.
