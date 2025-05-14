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
> Expand the list below to see which models support tensor parallelism. Open a GitHub issue or pull request to add support for a model not currently below.

<details>
<summary>Supported models</summary>

* [Cohere](./model_doc/cohere) and [Cohere 2](./model_doc/cohere2)
* [Gemma](./model_doc/gemma) and [Gemma 2](./model_doc/gemma2)
* [GLM](./model_doc/glm)
* [Granite](./model_doc/granite)
* [Llama](./model_doc/llama)
* [Mistral](./model_doc/mistral)
* [Mixtral](./model_doc/mixtral)
* [OLMo](./model_doc/olmo) and [OLMo2](./model_doc/olmo2)
* [Phi](./model_doc/phi) and [Phi-3](./model_doc/phi3)
* [Qwen2](./model_doc/qwen2), [Qwen2Moe](./model_doc/qwen2_moe), and [Qwen2-VL](./model_doc/qwen2_5_vl)
* [Starcoder2](./model_doc/starcoder2)

</details>

Set `tp_plan="auto"` in [`~AutoModel.from_pretrained`] to enable tensor parallelism for inference.

```py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# enable tensor parallelism
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    tp_plan="auto",
)

# prepare input tokens
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
prompt = "Can I help"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# distributed run
outputs = model(inputs)
```

Launch the inference script above on [torchrun](https://pytorch.org/docs/stable/elastic/run.html) with 4 processes per GPU.

```bash
torchrun --nproc-per-node 4 demo.py
```

For CPU, please binding different socket on each rank. For example, if you are using Intel 4th Gen Xeon:
```bash
export OMP_NUM_THREADS=56
numactl -C 0-55 -m 0 torchrun --nnodes=2 --node_rank=0 --master_addr="127.0.0.1" --master_port=29500 --nproc-per-node 1 demo.py & numactl -C 56-111 -m 1 torchrun --nnodes=2 --node_rank=1 --master_addr="127.0.0.1" --master_port=29500 --nproc-per-node 1 demo.py & wait
```
The CPU benchmark data will be released soon.

You can benefit from considerable speed ups for inference, especially for inputs with large batch size or long sequences.

For a single forward pass on [Llama](./model_doc/llama) with a sequence length of 512 and various batch sizes, you can expect the following speed ups.

<div style="text-align: center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Meta-Llama-3-8B-Instruct%2C%20seqlen%20%3D%20512%2C%20python%2C%20w_%20compile.png">
</div>
