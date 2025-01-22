<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Distributed GPU inference

[Tensor parallelism](./perf_train_gpu_many#tensor-parallelism) shards a model onto multiple GPUs and parallelizes computations such as matrix multiplication. It enables fitting larger model sizes into memory and is faster because each GPU can process a tensor slice.

> [!TIP]
> Tensor parallelism is currently only supported for [Llama](./model_doc/llama). Open a GitHub issue or pull request to add tensor parallelism support for another model.

Set `tp_plan="auto"` in [`~AutoModel.from_pretrained`] to enable tensor parallelism for inference.

```py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# initialize distributed environment
rank = int(os.environ["RANK"])
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)
torch.distributed.init_process_group("nccl", device_id=device)

# enable tensor parallelism
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    tp_plan="auto",
)

# prepare input tokens
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
prompt = "Can I help"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# distributed run
outputs = model(inputs)
```

Launch the inference script above on [torchrun](https://pytorch.org/docs/stable/elastic/run.html) with 4 processes per GPU.

```bash
torchrun --nproc-per-node 4 demo.py
```

You can benefit from considerable speed ups for inference, especially for inputs with large batch size or long sequences.

For a single forward pass on [Llama](./model_doc/llama) with a sequence length of 512 and various batch sizes, you can expect the following speed ups.

<div style="text-align: center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Meta-Llama-3-8B-Instruct%2C%20seqlen%20%3D%20512%2C%20python%2C%20w_%20compile.png">
</div>
