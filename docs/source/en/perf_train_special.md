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

Apple Silicon (M-series) chips have a unified memory architecture where the CPU and GPU share the same memory pool. Shared memory eliminates the data transfer overhead of GPUs, making it practical to train large models locally. Transformers uses the [Metal Performance Shaders (MPS)](https://pytorch.org/docs/stable/notes/mps.html) backend to use this architecture.

This requires macOS 12.3 or later and PyTorch built with MPS support.

> [!WARNING]
> MPS doesn't support all PyTorch operations yet. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall back to CPU kernels for unsupported operations. Open an issue in the [PyTorch](https://github.com/pytorch/pytorch/issues) repository for any other unexpected behavior.

## Device selection

[`Trainer`] detects MPS automatically with [torch.backends.mps.is_available](https://docs.pytorch.org/docs/main/notes/mps.html) and sets the device to `mps` without any configuration changes.

```python
training_args = TrainingArguments(
    output_dir="./outputs",
    # No extra flags needed — MPS is used automatically on Apple Silicon
)
```

## Mixed precision

MPS supports both bf16 and fp16 mixed precision. bf16 requires macOS 14.0 or later. It is not supported on earlier versions.

```python
training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,  # requires macOS 14.0+
)
```

## Model loading and device mapping

`device_map="auto"` treats MPS as a sole compute device without CPU offload. CUDA can spill layers to CPU RAM, but MPS requires the entire model to fit in unified memory.

If your model doesn't fit in unified memory, `device_map="auto"` won't work with MPS. Try a smaller model or lower precision instead.

## Metal quantization

Transformers includes an MPS-specific quantizer that uses Apple's Metal kernels for accelerated quantized inference. The Metal quantizer doesn't support training (`is_trainable = False`), so it won't reduce memory during a training run. It is useful for running inference on a quantized checkpoint after training.

To use it, install the `kernels` package:

```bash
pip install kernels
```

The Metal quantizer sets `device_map='mps'` automatically if you don't specify one. On-the-fly quantization with the Metal backend doesn't support CPU or disk offloading.

## Next steps

- Read the [Introducing Accelerated PyTorch Training on Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/) blog post for background on the MPS backend.
