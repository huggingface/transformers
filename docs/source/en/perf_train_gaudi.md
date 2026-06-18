<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Intel Gaudi

The Intel Gaudi AI accelerator family includes [Intel Gaudi 1](https://habana.ai/products/gaudi/), [Intel Gaudi 2](https://habana.ai/products/gaudi2/), and [Intel Gaudi 3](https://habana.ai/products/gaudi3/). Each server has 8 Habana Processing Units (HPUs) with 128GB of memory on Gaudi 3, 96GB on Gaudi 2, and 32GB on first-gen Gaudi. The [Gaudi Architecture](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html) overview covers the hardware in depth.

[`TrainingArguments`], [`Trainer`], and [`Pipeline`] detect Intel Gaudi devices and set the backend to `hpu` automatically.

## Environment variables

HPU lazy mode isn't compatible with all Transformers modeling code. Set the environment variable below to switch to eager mode if there are errors.

```bash
export PT_HPU_LAZY_MODE=0
```

You may also need to enable int64 support to avoid casting issues with long integers.

```bash
export PT_ENABLE_INT64_SUPPORT=1
```

## Mixed precision

All Gaudi generations support bf16 natively.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,  # supported on all Gaudi generations
)
```

## torch.compile

Gaudi supports [torch.compile](). [`TrainingArguments`] automatically sets `torch_compile_backend` to `"hpu_backend"` when HPU is detected.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    torch_compile=True,
)
```

## Distributed training

Multi-HPU training uses [HCCL](https://docs.habana.ai/en/latest/API_Reference_Guides/HCCL_APIs/index.html) (Habana Collective Communications Library) as the distributed backend. HCCL is the default, but you can also set `ddp_backend` explicitly.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    ddp_backend="hccl",
)
```

## Next steps

- See the [Gaudi docs](https://docs.habana.ai/en/latest/index.html) for more detailed information about training.
- Try [Optimum for Intel Gaudi](https://huggingface.co/docs/optimum/main/en/habana/index) for Gaudi-optimized model implementations during training and inference.
