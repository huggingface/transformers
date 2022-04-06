<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

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

# Semantic segmentation example

This directory contains a script, `run_semantic_segmentation_no_trainer.py`, that showcases how to fine-tune any `AutoModelForSemanticSegmentation` (such as SegFormer, BEiT, DPT) for semantic segmentation using PyTorch.

The script leverages [🤗 `Accelerate`](https://github.com/huggingface/accelerate), which allows to write your own training loop in PyTorch, but have it run instantly on any (distributed) environment, including CPU, multi-CPU, GPU, multi-GPU and TPU. It also supports mixed precision. 

First, run:

```bash
accelerate config
```

and reply to the questions asked regarding the environment on which you'd like to train. Then:

```bash
accelerate launch run_semantic_segmentation_no_trainer.py 
```

and boom, you're training, possible on multiple GPUs :) 