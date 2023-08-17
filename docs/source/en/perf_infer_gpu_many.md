<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Efficient Inference on a Multiple GPUs

This document contains information on how to efficiently infer on a multiple GPUs. 
<Tip>

Note: A multi GPU setup can use the majority of the strategies described in the [single GPU section](./perf_infer_gpu_one). You must be aware of simple techniques, though, that can be used for a better usage.

</Tip>

## BetterTransformer

[`BetterTransformer`](https://huggingface.co/docs/optimum/bettertransformer/overview) API converts 🤗 transformers models to make them use PyTorch-native transformer fastpath that calls optimized kernels such as Flash Attention under the hood.  

BetterTransformer is also supported for faster inference on multi-GPU for text, image, and audio models.

<Tip>

Flash Attention can only be used for models using fp16 or bf16 dtype. Make sure to cast your model before using BetterTransformer.
  
</Tip>

### Decoder models

For text models, especially decoder-based models (GPT, T5, Llama, etc.), the `BetterTransformer` API converts all attention operations to use the [`torch.nn.functional.scaled_dot_product_attention` operator](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention) (SDPA) that is only available in PyTorch 2.0 and onwards. 

To convert a model to `BetterTransformer`:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
# convert the model to BetterTransformer
model.to_bettertransformer()

# Use it for training or inference
```

SDPA can also call [Flash-Attention](https://arxiv.org/abs/2205.14135) kernels under the hood. In order to force the usage of Flash Attention or check that Flash Attention is available in a given setting (hardware, problem size), the context manager [`with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)`](https://pytorch.org/docs/master/backends.html#torch.backends.cuda.sdp_kernel) can be used:


```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m").to("cuda")
# convert the model to BetterTransformer
model.to_bettertransformer()

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

If you see a bug with a traceback saying 

```bash
RuntimeError: No available kernel.  Aborting execution.
```

you may try to use the PyTorch nightly version, which may have a larger coverage for Flash Attention:

```bash
pip3 install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118


Have a look at this [blog post](https://pytorch.org/blog/out-of-the-box-acceleration/) to learn more about what is possible to do with the `BetterTransformer` + SDPA API.

### Encoder models

For encoder models during inference, BetterTransformer will dispatch the forward call of encoder layers to an equivalent of [`torch.nn.TransformerEncoderLayer`](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) that will execute the fast path implementation of the encoder layers.

As `torch.nn.TransformerEncoderLayer` fastpath does not support training, we instead dispatch to `torch.nn.functional.scaled_dot_product_attention` during training, that do not leverage nested tensors but can leverage Flash Attention or Memory-Efficient Attention fused kernels.

More details can be found in [this blog post](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2) and [this second one](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/).