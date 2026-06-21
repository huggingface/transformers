<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Apple Silicon

Apple Silicon (M-series) chips have a unified memory architecture where the CPU and GPU share the same memory pool. Shared memory eliminates the data transfer overhead of GPUs, making it practical to train large models locally. Transformers uses the [Metal Performance Shaders (MPS)](https://pytorch.org/docs/stable/notes/mps.html) backend to accelerate training on this hardware.

This requires macOS 12.3 or later and PyTorch built with MPS support.

> [!WARNING]
> MPS doesn't support all PyTorch operations yet (see this [GitHub issue](https://github.com/pytorch/pytorch/issues/77764) for more details about missing ops). Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall back to CPU kernels for unsupported operations. Open an issue in the [PyTorch](https://github.com/pytorch/pytorch/issues) repository for any other unexpected behavior.

## Model loading and device selection

MPS requires the entire model to fit in unified memory, so `device_map="auto"` can't offload layers to the CPU like CUDA. In this case, try using a smaller model.

[`Trainer`] detects MPS automatically with `torch.backends.mps.is_available` and sets the device to `mps` without any configuration changes.

## Mixed precision

MPS supports both bf16 and fp16 mixed precision (bf16 requires macOS 14.0 or later).

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,  # requires macOS 14.0+
)
```

## Next steps

- Read the [Introducing Accelerated PyTorch Training on Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/) blog post for background on the MPS backend.
