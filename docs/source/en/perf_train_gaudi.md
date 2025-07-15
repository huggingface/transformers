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

The Intel Gaudi AI accelerator family includes [Intel Gaudi 1](https://habana.ai/products/gaudi/), [Intel Gaudi 2](https://habana.ai/products/gaudi2/), and [Intel Gaudi 3](https://habana.ai/products/gaudi3/). Each server is equipped with 8 devices, known as Habana Processing Units (HPUs), providing 128GB of memory on Gaudi 3, 96GB on Gaudi 2, and 32GB on the first-gen Gaudi. For more details on the underlying hardware architecture, check out the [Gaudi Architecture](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html) overview.

[`TrainingArguments`], [`Trainer`] and [`Pipeline`] detect and set the backend device to `hpu` if an Intel Gaudi device is available. No additional changes are required to enable training and inference on your device.

Some modeling code in Transformers is not optimized for HPU lazy mode. If you encounter any errors, set the environment variable below to use eager mode:
```
PT_HPU_LAZY_MODE=0
```

In some cases, you'll also need to enable int64 support to avoid casting issues with long integers:
```
PT_ENABLE_INT64_SUPPORT=1
```
Refer to the [Gaudi docs](https://docs.habana.ai/en/latest/index.html) for more details.

> [!TIP]
> For training and inference with Gaudi-optimized model implementations, we recommend using [Optimum for Intel Gaudi](https://huggingface.co/docs/optimum/main/en/habana/index).
