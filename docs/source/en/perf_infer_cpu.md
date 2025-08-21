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

### BetterTransformer

[BetterTransformer](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/) is a *fastpath* execution of specialized Transformers functions directly on the hardware level such as a CPU. There are two main components of the fastpath execution.

- fusing multiple operations into a single kernel for faster and more efficient execution
- skipping unnecessary computation of padding tokens with nested tensors

> [!WARNING]
> BetterTransformer isn't supported for all models. Check this [list](https://hf.co/docs/optimum/bettertransformer/overview#supported-models) to see whether a model supports BetterTransformer.

BetterTransformer is available through Optimum with [`~PreTrainedModel.to_bettertransformer`].

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom")
model = model.to_bettertransformer()
```

## TorchScript

[TorchScript](https://pytorch.org/docs/stable/jit.html) is an intermediate PyTorch model format that can be run in non-Python environments, like C++, where performance is critical. Train a PyTorch model and convert it to a TorchScript function or module with [torch.jit.trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html). This function optimizes the model with just-in-time (JIT) compilation, and compared to the default eager mode, JIT-compiled models offer better inference performance.

> [!TIP]
> Refer to the [Introduction to PyTorch TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) tutorial for a gentle introduction to TorchScript.

On a CPU, enable `torch.jit.trace` with the `--jit_mode_eval` flag in [`Trainer`].

```bash
python examples/pytorch/question-answering/run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
--jit_mode_eval
```
