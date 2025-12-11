<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Optimum

[Optimum](https://huggingface.co/docs/optimum/main/en/index) is a hardware-aware optimization library designed to help a model run efficiently on whatever hardware or framework you're using. It supports [ONNX Runtime (ORT)](https://huggingface.co/docs/optimum-onnx/index), a model accelerator that takes a ONNX-formatted model and executes it with highly optimized kernels depending on the best backend available on your machine.

This guide shows how to inference with Optimum's [`~optimum.onnxruntime.ORTModel`].

## GPU

Select a provider below based on the GPU you're using.

| provider | hardware |
|---|---|
| [CUDAExecutionProvider](https://hf.co/docs/optimum/main/en/onnxruntime/usage_guides/gpu#cudaexecutionprovider) | CUDA-enabled GPUs |
| [ROCMExecutionProvider](https://hf.co/docs/optimum/onnxruntime/usage_guides/amdgpu) | AMD Instinct, Radeon Pro, Radeon GPUs |
| [TensorrtExecutionProvider](https://hf.co/docs/optimum/onnxruntime/usage_guides/gpu#tensorrtexecutionprovider) | TensorRT |

Pass it to the `provider` argument in [`~optimum.onnxruntime.ORTModel.from_pretrained`] to enable your hardware's specific optimizations.

```py
from optimum.onnxruntime import ORTModelForCausalLM

ort_model = ORTModelForCausalLM.from_pretrained(
  "HuggingFaceTB/SmolLM-135M",
  provider="CUDAExecutionProvider",
)
```

Set `export=True` if a checkpoint doesn't have a `model.onnx` file. This converts the checkpoint on the fly.

```py
from optimum.onnxruntime import ORTModelForCausalLM

ort_model = ORTModelForCausalLM.from_pretrained(
  "HuggingFaceTB/SmolLM-135M",
  export=True,
  provider="CUDAExecutionProvider",
)
```

## CPU

The `provider` argument defaults to `"CPUExecutionProvider"` so you don't need to explicitly set it if you're using a CPU.

```py
from optimum.onnxruntime import ORTModelForCausalLM

ort_model = ORTModelForCausalLM.from_pretrained(
  "HuggingFaceTB/SmolLM-135M",
)
```

Optimum provides an [Intel](https://hf.co/docs/optimum/intel/index) extension that provides additional optimizations such as quantization, pruning, and knowledge distillation for Intel CPUs. This extension also includes tools to convert models to [OpenVINO](https://hf.co/docs/optimum/intel/inference), a toolkit for optimizing and deploying models, for even faster inference.