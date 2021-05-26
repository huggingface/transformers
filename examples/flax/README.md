<!---
Copyright 2021 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# JAX/Flax Examples

This folder contains actively maintained examples of 🤗 Transformers using the JAX/Flax backend. Porting models and examples to JAX/Flax is an ongoing effort, and more will be added in the coming months. In particular, these examples are all designed to run fast on Cloud TPUs, and we include step-by-step guides to getting started with Cloud TPU.

*NOTE*: Currently, there is no "Trainer" abstraction for JAX/Flax -- all examples contain an explicit training loop.

## Intro: JAX and Flax

[JAX](https://github.com/google/jax) is a numerical computation library that exposes a NumPy-like API with tracing capabilities. With JAX's `jit`, you can
trace pure functions and compile them into efficient, fused accelerator code on both GPU and TPU. JAX
supports additional transformations such as `grad` (for arbitrary gradients), `pmap` (for parallelizing computation on multiple devices), `remat` (for gradient checkpointing), `vmap` (automatic
efficient vectorization), and `pjit` (for automatically sharded model parallelism). All JAX transformations compose arbitrarily with each other -- e.g., efficiently
computing per-example gradients is simply `vmap(grad(f))`.

[Flax](https://github.com/google/flax) builds on top of JAX with an ergonomic
module abstraction using Python dataclasses that leads to concise and explicit code. Flax's "lifted" JAX transformations (e.g. `vmap`, `remat`) allow you to nest JAX transformation and modules in any way you wish. Flax is the most widely used JAX library, with [129 dependent projects](https://github.com/google/flax/network/dependents?package_id=UGFja2FnZS01MjEyMjA2MA%3D%3D) as of May 2021. It is also the library underlying all of the official Cloud TPU JAX examples.

## Running on Cloud TPU

All of our JAX/Flax models are designed to run efficiently on Google
Cloud TPUs. Here is [a guide for running JAX on Google Cloud TPU](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm).

Each example README contains more details on the specific model and training
procedure.

## Supported models

Porting models from PyTorch to JAX/Flax is an ongoing effort. 
Feel free to reach out if you are interested in contributing a model in JAX/Flax -- we'll 
be adding a guide for porting models from PyTorch in the upcoming few weeks.

For a complete overview of models that are supported in JAX/Flax, please have a look at [this](https://huggingface.co/transformers/master/index.html#supported-frameworks) table.

Over 3000 pretrained checkpoints are supported in JAX/Flax as of May 2021.
Click [here](https://huggingface.co/models?filter=jax) to see the full list on the 🤗 hub. 

## Examples

The following table lists all of our examples on how to use 🤗 Transformers with the JAX/Flax backend:
- with information about the model and dataset used,
- whether or not they leverage the [🤗 Datasets](https://github.com/huggingface/datasets) library,
- links to **Colab notebooks** to walk through the scripts and run them easily.

| Task | Example model | Example dataset | 🤗 Datasets | Colab
|---|---|---|:---:|:---:|
| [**`masked-language-modeling`**](https://github.com/huggingface/transformers/tree/master/examples/flax/language-modeling) | BERT | OSCAR | ✅ | [![Open In Colab (TODO: Patrick)](https://colab.research.google.com/assets/colab-badge.svg)]()
| [**`text-classification`**](https://github.com/huggingface/transformers/tree/master/examples/flax/text-classification) | BERT | GLUE | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb)
