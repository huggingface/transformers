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

Padding-free training (also called packing) concatenates several samples into a single sequence instead of padding each one to a fixed length. The model needs to know where each sample ends so (linear) attention doesn't mix tokens across samples.

There are two ways to provide those boundaries.

- Prepare them ahead of time with a data collator.
- Infer them from `position_ids` at runtime.

The recommended approach is the data collator. This guide explains why and covers the caveats of the `position_ids` path.

> [!WARNING]
> Inferring boundaries from `position_ids` is not the preferred approach, and it only works for standard attention models. Linear-attention models such as Qwen3-Next and Qwen3.5 (Gated DeltaNet) and convolution-based models ignore `position_ids` boundaries and require the data collator. See [Linear attention and convolution models](#linear-attention-and-convolution-models).

## Prepare boundaries with a data collator

Preparing the boundary kwargs up front removes the problems above and behaves identically whether or not you compile.

Use [`DataCollatorWithFlattening`] to flatten each batch and return the boundary information. Set `return_flash_attn_kwargs=True` so the collator precomputes the boundaries instead of leaving them to be inferred from `position_ids` at runtime. Pass it to [`Trainer`] and don't add an `attention_mask`, since the flattened batch already encodes the boundaries and a mask conflicts with the packed layout.

> [!TIP]
> Padding-free relies on a FlashAttention implementation for standard attention models, since only the FlashAttention kernels expose the variable-length path that a flattened batch needs.
>
> Install the [kernels](./kernels) library, which fetches a prebuilt FlashAttention kernel without requiring a local build. It also works as a fallback when [flash-attn](https://github.com/Dao-AILab/flash-attention) isn't installed locally. Load the model with `attn_implementation="kernels-community/flash-attn2"`.

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithFlattening, Trainer, TrainingArguments

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
    device_map="auto",
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

## Infer boundaries from position_ids

FlashAttention can detect padding-free batches from `position_ids` alone and remains for backward compatibility, because downstream frameworks such as TRL depend on it.

Relying on `position_ids` has two problems.

- Detecting packed sequences from `position_ids` is a dynamic, data-dependent check. It works without compilation, but under `torch.compile` it causes graph breaks. The check is currently restricted to `batch_size == 1` to limit how often it runs, since real batch sizes are usually larger.
- Compiled FlashAttention forces some kwargs to be plain Python `int`s. Inferring them from `position_ids` at runtime forces device-to-host syncs, and on older PyTorch versions an extra graph break from the tensor-to-int conversion.

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

The exact kernel packages depend on the model's original implementation. Gated DeltaNet models such as Qwen3-Next and Qwen3.5 use [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), and Mamba-based models such as Bamba use [mamba-ssm](https://github.com/state-spaces/mamba). Both rely on [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) for the convolution. Without the right kernels, the model falls back to reference implementations that ignore the boundary kwargs and mix tokens across samples.

> [!TIP]
> Many of these kernels are also available through the [kernels](./kernels) library, which can fetch a compatible build for you. flash-linear-attention typically still needs a direct install.

When the boundary kwargs are missing, the kernels quietly treat the whole batch as one sequence. Nothing raises an error or warning, because a runtime check would add a data-dependent branch that conflicts with `torch.compile`.

## Next steps

- See the [data collators](./data_collators) guide for other collators.
- Browse the [`DataCollatorWithFlattening`] API reference for the full set of arguments.
- Read [Improving Hugging Face Training Efficiency Through Packing with Flash Attention](https://huggingface.co/blog/packing-with-FA2) for benchmarks and a deeper walkthrough.
