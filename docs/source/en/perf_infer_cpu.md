<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CPU

CPUs are a viable and cost-effective inference option. With a few optimization methods, it is possible to achieve good performance with large models on CPUs. These methods include fusing kernels to reduce overhead and compiling your code to a faster intermediate format that can be deployed in production environments.

This guide will show you a few ways to optimize inference on a CPU.

## Optimum

[Optimum](https://hf.co/docs/optimum/en/index) is a Hugging Face library focused on optimizing model performance across various hardware. It supports [ONNX Runtime](https://onnxruntime.ai/docs/) (ORT), a model accelerator, for a wide range of hardware and frameworks including CPUs.

Optimum provides the [`~optimum.onnxruntime.ORTModel`] class for loading ONNX models. For example, load the [optimum/roberta-base-squad2](https://hf.co/optimum/roberta-base-squad2) checkpoint for question answering inference. This checkpoint contains a [model.onnx](https://hf.co/optimum/roberta-base-squad2/blob/main/model.onnx) file.

```py
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForQuestionAnswering

onnx_qa = pipeline("question-answering", model="optimum/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")

question = "What's my name?"
context = "My name is Philipp and I live in Nuremberg."
pred = onnx_qa(question, context)
```

> [!TIP]
> Optimum includes an [Intel](https://hf.co/docs/optimum/intel/index) extension that provides additional optimizations such as quantization, pruning, and knowledge distillation for Intel CPUs. This extension also includes tools to convert models to [OpenVINO](https://hf.co/docs/optimum/intel/inference), a toolkit for optimizing and deploying models, for even faster inference.
