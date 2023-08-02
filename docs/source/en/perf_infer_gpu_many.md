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

## `BetterTransformer` API and Flash Attention for faster inference

We have recently integrated `BetterTransformer` for faster inference on multi-GPU for text, image and audio models. Check the documentation about this integration [here](https://huggingface.co/docs/optimum/bettertransformer/overview) for more details. 

For text models, especially decoder-based models (e.g. GPT, T5, Llama, etc.), the `BetterTransformer` API will convert all attention operations to use the [`torch.nn.functional.scaled_dot_product_attention` method](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention) (SDPA), that is available only from PyTorch 2.0 and onwards. 

An example usage of the `BetterTransformer` API is shown below:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
# convert the model to BetterTransformer
model.to_bettertransformer()

# Use it for training or inference
```

According to the official documentation, `torch.nn.functional.scaled_dot_product_attention` can also call [Flash-Attention](https://arxiv.org/abs/2205.14135) kernels under the hood. If you want to force the usage of Flash Attention, you can use the [`torch.backends.cuda.sdp_kernel(enable_flash=True)`](https://pytorch.org/docs/master/backends.html#torch.backends.cuda.sdp_kernel) as below:


```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m").to("cuda")
# convert the model to BetterTransformer
model.to_bettertransformer()

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

with torch.backends.cuda.sdp_kernel(enable_flash=True):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

For encoder models, the method [`~PreTrainedModel.reverse_bettertransformer`] allows to go back to the original modeling, which should be used before saving the model in order to use the canonical transformers modeling:

```python
model = model.reverse_bettertransformer()
model.save_pretrained("saved_model")
```

Have a look at [this detailed blogpost](https://pytorch.org/blog/out-of-the-box-acceleration/) to read more about what is possible to do with `BetterTransformer` + SDPA API.