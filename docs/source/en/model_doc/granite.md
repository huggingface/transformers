<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->


<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
</div>

# Granite

[Granite](https://huggingface.co/papers/2408.13359) is a 3B parameter language model trained with the Power scheduler. Discovering a good learning rate for pretraining large language models is difficult because it depends on so many variables (batch size, number of training tokens, etc.) and it is expensive to perform a hyperparameter search. The Power scheduler is based on a power-law relationship between the variables and their transferability to larger models. Combining the Power scheduler with Maximum Update Parameterization (MUP) allows a model to be pretrained with one set of hyperparameters regardless of all the variables.

You can find all the original Granite checkpoints under the [IBM-Granite](https://huggingface.co/ibm-granite) organization.

> [!TIP]
> Click on the Granite models in the right sidebar for more examples of how to apply Granite to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`, and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="ibm-granite/granite-3.3-2b-base",
    dtype=torch.bfloat16,
    device=0
)
pipe("Explain quantum computing in simple terms ", max_new_tokens=50)
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-base")
model = AutoModelForCausalLM.from_pretrained(
    "ibm-granite/granite-3.3-2b-base",                                          
    dtype=torch.bfloat16, 
    device_map="auto",
    attn_implementation="sdpa"
)

inputs = tokenizer("Explain quantum computing in simple terms", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=50, cache_implementation="static")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
</hfoption>
<hfoption id="transformers CLI">

```python
echo -e "Explain quantum computing simply." | transformers-cli run --task text-generation --model ibm-granite/granite-3.3-8b-instruct --device 0
```
</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to int4.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-8b-base")
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.3-8b-base", dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa", quantization_config=quantization_config)

inputs = tokenizer("Explain quantum computing in simple terms", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=50, cache_implementation="static")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(""ibm-granite/granite-3.3-2b-base"")
model = AutoModelForCausalLM.from_pretrained(
    "ibm-granite/granite-3.3-2b-base",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
    quantization_config=quantization_config,
)

input_ids = tokenizer("Explain artificial intelligence to a 10 year old", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=50, cache_implementation="static")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

  
## GraniteConfig

[[autodoc]] GraniteConfig

## GraniteModel

[[autodoc]] GraniteModel
    - forward

## GraniteForCausalLM

[[autodoc]] GraniteForCausalLM
    - forward
