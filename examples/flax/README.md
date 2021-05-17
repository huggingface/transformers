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

This folder contains actively maintained examples of use of 🤗 Transformers using the JAX/Flax backend. Porting models and examples to JAX/Flax is an ongoing effort, and more will be added in the coming months. In particular, these examples are all designed to run fast on Cloud TPUs, and we include step-by-step guides to getting started with Cloud TPU.

*NOTE*: There current is no "Trainer" abstraction for JAX/Flax -- all examples contain an explicit training loop.

## Intro: JAX and Flax

[JAX](https://github.com/google/jax) is a numerical computation library that exposes a NumPy-like API with tracing capabilities. With JAX's `jit`, you can
trace pure functions and compile them into efficient, fused accelerator code on both GPU and TPU. JAX
supports additional transformations such as `grad` (for arbitrary gradients), `pmap` (for parallelizing computation on multiple devices), `remat` (for gradient checkpointing), `vmap` (automatic
efficient vectorization), and `pjit` (for automatically sharded model parallelism). All JAX transformations compose arbitrarily with each other -- e.g., efficiently
computing per-example gradients is simply `vmap(grad(f))`.

[Flax](https://github.com/google/flax) builds on top of JAX with an ergonomic
module abstraction using Python dataclasses that leads to concise and explicit code. Flax's "lifted" JAX transformations (e.g. `vmap`, `remat`) allow you to nest JAX transformation and modules in any way you wish. Flax is the most widely used JAX library, with [129 dependent projects](https://github.com/google/flax/network/dependents?package_id=UGFja2FnZS01MjEyMjA2MA%3D%3D) as of May 2021. It is also the library underlying all of the official Cloud TPU JAX examples. (TODO: Add link once it's there.)

## Running on Cloud TPU

All of our JAX/Flax models are designed to run efficiently on Google
Cloud TPUs. Here is a guide for running jobs on Google Cloud TPU.
(TODO: Add a link to the Cloud TPU JAX getting started guide once it's public)
Each example README contains more details on the specific model and training
procedure in question.

## Ported models

* [BERT](../../src/transformers/models/bert/modeling_flax_bert.py)
* [ELECTRA](../../src/transformers/models/electra/modeling_flax_electra.py)
* [RoBERTa](../../src/transformers/models/roberta/modeling_flax_roberta.py)

(See the full list here: https://huggingface.co/transformers/)

## Pre-training examples

* [Masked Language Modeling](./language-modeling) (TODO: Add README)

## Fine-tuning examples

* [GLUE](./text-classification)

