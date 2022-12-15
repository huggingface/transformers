<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
-->

# Efficient Inference on CPU

This guide focuses on inferencing large models efficiently on CPU.

## `BetterTransformer` for faster inference

We have recently integrated `BetterTransformer` for faster inference on CPU for text, image and audio models. Check the documentation about this integration [here](https://huggingface.co/docs/optimum/bettertransformer/overview) for more details.

## PyTorch JIT-mode (TorchScript)
TorchScript is a way to create serializable and optimizable models from PyTorch code. Any TorchScript program can be saved from a Python process and loaded in a process where there is no Python dependency.
Comparing to default eager mode, jit mode in PyTorch normally yields better performance for model inference from optimization methodologies like operator fusion.

For a gentle introduction to TorchScript, see the Introduction to [PyTorch TorchScript tutorial](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#tracing-modules).

### IPEX Graph Optimization with JIT-mode
Intel® Extension for PyTorch provides further optimizations in jit mode for Transformers series models. It is highly recommended for users to take advantage of Intel® Extension for PyTorch with jit mode. Some frequently used operator patterns from Transformers models are already supported in Intel® Extension for PyTorch with jit mode fusions. Those fusion patterns like Multi-head-attention fusion, Concat Linear, Linear+Add, Linear+Gelu, Add+LayerNorm fusion and etc. are enabled and perform well. The benefit of the fusion is delivered to users in a transparent fashion. According to the analysis, ~70% of most popular NLP tasks in question-answering, text-classification, and token-classification can get performance benefits with these fusion patterns for both Float32 precision and BFloat16 Mixed precision.

Check more detailed information for [IPEX Graph Optimization](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/graph_optimization.html).

#### IPEX installation:

IPEX release is following PyTorch, check the approaches for [IPEX installation](https://intel.github.io/intel-extension-for-pytorch/).

### Usage of JIT-mode
To enable JIT-mode in Trainer for evaluaion or prediction, users should add `jit_mode_eval` in Trainer command arguments.

<Tip warning={true}>

for PyTorch >= 1.14.0. JIT-mode could benefit any models for prediction and evaluaion since dict input is supported in jit.trace

for PyTorch < 1.14.0. JIT-mode could benefit models whose forward parameter order matches the tuple input order in jit.trace, like question-answering model
In the case where the forward parameter order does not match the tuple input order in jit.trace, like text-classification models, jit.trace will fail and we are capturing this with the exception here to make it fallback. Logging is used to notify users.

</Tip>

Take an example of the use cases on [Transformers question-answering](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)


- Inference using jit mode on CPU:
<pre>python run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
<b>--jit_mode_eval </b></pre> 

- Inference with IPEX using jit mode on CPU:
<pre>python run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
<b>--use_ipex \</b>
<b>--jit_mode_eval</b></pre> 
