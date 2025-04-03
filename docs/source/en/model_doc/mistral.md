<!--Copyright 2023 Mistral AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Mistral

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
<img alt="Flax" src="https://img.shields.io/badge/Flax-29a79b.svg?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAtCAMAAAANxBKoAAAC7lBMVEUAAADg5vYHPVgAoJH+/v76+v39/f9JbLP///9+AIgAnY3///+mcqzt8fXy9fgkXa3Ax9709fr+///9/f8qXq49qp5AaLGMwrv8/P0eW60VWawxYq8yqJzG2dytt9Wyu9elzci519Lf3O3S2efY3OrY0+Xp7PT///////+dqNCexMc6Z7AGpJeGvbenstPZ5ejQ1OfJzOLa7ejh4+/r8fT29vpccbklWK8PVa0AS6ghW63O498vYa+lsdKz1NDRt9Kw1c672tbD3tnAxt7R6OHp5vDe7OrDyuDn6vLl6/EAQKak0MgATakkppo3ZK/Bz9y8w9yzu9jey97axdvHzeG21NHH4trTwthKZrVGZLSUSpuPQJiGAI+GAI8SWKydycLL4d7f2OTi1+S9xNzL0ePT6OLGzeEAo5U0qJw/aLEAo5JFa7JBabEAp5Y4qZ2QxLyKmsm3kL2xoMOehrRNb7RIbbOZgrGre68AUqwAqZqNN5aKJ5N/lMq+qsd8kMa4pcWzh7muhLMEV69juq2kbKqgUaOTR5uMMZWLLZSGAI5VAIdEAH+ovNDHuNCnxcy3qcaYx8K8msGplrx+wLahjbYdXrV6vbMvYK9DrZ8QrZ8tqJuFms+Sos6sw8ecy8RffsNVeMCvmb43aLltv7Q4Y7EZWK4QWa1gt6meZKUdr6GOAZVeA4xPAISyveLUwtivxtKTpNJ2jcqfvcltiMiwwcfAoMVxhL+Kx7xjdrqTe60tsaNQs6KaRKACrJ6UTZwkqpqTL5pkHY4AloSgsd2ptNXPvNOOncuxxsqFl8lmg8apt8FJcr9EbryGxLqlkrkrY7dRa7ZGZLQ5t6iXUZ6PPpgVpZeJCJFKAIGareTa0+KJod3H0deY2M+esM25usmYu8d2zsJOdcBVvrCLbqcAOaaHaKQAMaScWqKBXqCXMJ2RHpiLF5NmJZAdAHN2kta11dKu1M+DkcZLdb+Mcql3TppyRJdzQ5ZtNZNlIY+DF4+voCOQAAAAZ3RSTlMABAT+MEEJ/RH+/TP+Zlv+pUo6Ifz8+fco/fz6+evr39S9nJmOilQaF/7+/f38+smmoYp6b1T+/v7++vj189zU0tDJxsGzsrKSfv34+Pf27dDOysG9t6+n/vv6+vr59uzr1tG+tZ6Qg9Ym3QAABR5JREFUSMeNlVVUG1EQhpcuxEspXqS0SKEtxQp1d3d332STTRpIQhIISQgJhODu7lAoDoUCpe7u7u7+1puGpqnCPOyZvffbOXPm/PsP9JfQgyCC+tmTABTOcbxDz/heENS7/1F+9nhvkHePG0wNDLbGWwdXL+rbLWvpmZHXD8+gMfBjTh+aSe6Gnn7lwQIOTR0c8wfX3PWgv7avbdKwf/ZoBp1Gp/PvuvXW3vw5ib7emnTW4OR+3D4jB9vjNJ/7gNvfWWeH/TO/JyYrsiKCRjVEZA3UB+96kON+DxOQ/NLE8PE5iUYgIXjFnCOlxEQMaSGVxjg4gxOnEycGz8bptuNjVx08LscIgrzH3umcn+KKtiBIyvzOO2O99aAdR8cF19oZalnCtvREUw79tCd5sow1g1UKM6kXqUx4T8wsi3sTjJ3yzDmmhenLXLpo8u45eG5y4Vvbk6kkC4LLtJMowkSQxmk4ggVJEG+7c6QpHT8vvW9X7/o7+3ELmiJi2mEzZJiz8cT6TBlanBk70cB5GGIGC1gRDdZ00yADLW1FL6gqhtvNXNG5S9gdSrk4M1qu7JAsmYshzDS4peoMrU/gT7qQdqYGZaYhxZmVbGJAm/CS/HloWyhRUlknQ9KYcExTwS80d3VNOxUZJpITYyspl0LbhArhpZCD9cRWEQuhYkNGMHToQ/2Cs6swJlb39CsllxdXX6IUKh/H5jbnSsPKjgmoaFQ1f8wRLR0UnGE/RcDEjj2jXG1WVTwUs8+zxfcrVO+vSsuOpVKxCfYZiQ0/aPKuxQbQ8lIz+DClxC8u+snlcJ7Yr1z1JPqUH0V+GDXbOwAib931Y4Imaq0NTIXPXY+N5L18GJ37SVWu+hwXff8l72Ds9XuwYIBaXPq6Shm4l+Vl/5QiOlV+uTk6YR9PxKsI9xNJny31ygK1e+nIRC1N97EGkFPI+jCpiHe5PCEy7oWqWSwRrpOvhFzcbTWMbm3ZJAOn1rUKpYIt/lDhW/5RHHteeWFN60qo98YJuoq1nK3uW5AabyspC1BcIEpOhft+SZAShYoLSvnmSfnYADUERP5jJn2h5XtsgCRuhYQqAvwTwn33+YWEKUI72HX5AtfSAZDe8F2DtPPm77afhl0EkthzuCQU0BWApgQIH9+KB0JhopMM7bJrdTRoleM2JAVNMyPF+wdoaz+XJpGoVAQ7WXUkcV7gT3oUZyi/ISIJAVKhgNp+4b4veCFhYVJw4locdSjZCp9cPUhLF9EZ3KKzURepMEtCDPP3VcWFx4UIiZIklIpFNfHpdEafIF2aRmOcrUmjohbT2WUllbmRvgfbythbQO3222fpDJoufaQPncYYuqoGtUEsCJZL6/3PR5b4syeSjZMQG/T2maGANlXT2v8S4AULWaUkCxfLyW8iW4kdka+nEMjxpL2NCwsYNBp+Q61PF43zyDg9Bm9+3NNySn78jMZUUkumqE4Gp7JmFOdP1vc8PpRrzj9+wPinCy8K1PiJ4aYbnTYpCCbDkBSbzhu2QJ1Gd82t8jI8TH51+OzvXoWbnXUOBkNW+0mWFwGcGOUVpU81/n3TOHb5oMt2FgYGjzau0Nif0Ss7Q3XB33hjjQHjHA5E5aOyIQc8CBrLdQSs3j92VG+3nNEjbkbdbBr9zm04ruvw37vh0QKOdeGIkckc80fX3KH/h7PT4BOjgCty8VZ5ux1MoO5Cf5naca2LAsEgehI+drX8o/0Nu+W0m6K/I9gGPd/dfx/EN/wN62AhsBWuAAAAAElFTkSuQmCC
">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Mistral was introduced in the [this blogpost](https://mistral.ai/news/announcing-mistral-7b/) by Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed.

The introduction of the blog post says:

*Mistral AI team is proud to release Mistral 7B, the most powerful language model for its size to date.*

Mistral-7B is the first large language model (LLM) released by [mistral.ai](https://mistral.ai/).

### Architectural details

Mistral-7B is a decoder-only Transformer with the following architectural choices:

- Sliding Window Attention - Trained with 8k context length and fixed cache size, with a theoretical attention span of 128K tokens
- GQA (Grouped Query Attention) - allowing faster inference and lower cache size.
- Byte-fallback BPE tokenizer - ensures that characters are never mapped to out of vocabulary tokens.

For more details refer to the [release blog post](https://mistral.ai/news/announcing-mistral-7b/).

### License

`Mistral-7B` is released under the Apache 2.0 license.

## Usage tips

The Mistral team has released 3 checkpoints:

- a base model, [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), which has been pre-trained to predict the next token on internet-scale data.
- an instruction tuned model, [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1), which is the base model optimized for chat purposes using supervised fine-tuning (SFT) and direct preference optimization (DPO).
- an improved instruction tuned model, [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), which improves upon v1.

The base model can be used as follows:

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

>>> prompt = "My favourite condiment is"

>>> model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
>>> model.to(device)

>>> generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"My favourite condiment is to ..."
```

The instruction tuned model can be used as follows:

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

>>> messages = [
...     {"role": "user", "content": "What is your favourite condiment?"},
...     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
...     {"role": "user", "content": "Do you have mayonnaise recipes?"}
... ]

>>> model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

>>> generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"Mayonnaise can be made as follows: (...)"
```

As can be seen, the instruction-tuned model requires a [chat template](../chat_templating) to be applied to make sure the inputs are prepared in the right format.

## Speeding up Mistral by using Flash Attention

The code snippets above showcase inference without any optimization tricks. However, one can drastically speed up the model by leveraging [Flash Attention](../perf_train_gpu_one#flash-attention-2), which is a faster implementation of the attention mechanism used inside the model.

First, make sure to install the latest version of Flash Attention 2 to include the sliding window attention feature.

```bash
pip install -U flash-attn --no-build-isolation
```

Make also sure that you have a hardware that is compatible with Flash-Attention 2. Read more about it in the official documentation of the [flash attention repository](https://github.com/Dao-AILab/flash-attention). Make also sure to load your model in half-precision (e.g. `torch.float16`)

To load and run a model using Flash Attention-2, refer to the snippet below:

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

>>> prompt = "My favourite condiment is"

>>> model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
>>> model.to(device)

>>> generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"My favourite condiment is to (...)"
```

### Expected speedups

Below is a expected speedup diagram that compares pure inference time between the native implementation in transformers using `mistralai/Mistral-7B-v0.1` checkpoint and the Flash Attention 2 version of the model.

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/mistral-7b-inference-large-seqlen.png">
</div>

### Sliding window Attention

The current implementation supports the sliding window attention mechanism and memory efficient cache management. 
To enable sliding window attention, just make sure to have a `flash-attn` version that is compatible with sliding window attention (`>=2.3.0`). 

The Flash Attention-2 model uses also a more memory efficient cache slicing mechanism - as recommended per the official implementation of Mistral model that use rolling cache mechanism we keep the cache size fixed (`self.config.sliding_window`), support batched generation only for `padding_side="left"` and use the absolute position of the current token to compute the positional embedding.

## Shrinking down Mistral using quantization

As the Mistral model has 7 billion parameters, that would require about 14GB of GPU RAM in half precision (float16), since each parameter is stored in 2 bytes. However, one can shrink down the size of the model using [quantization](../quantization.md). If the model is quantized to 4 bits (or half a byte per parameter),that requires only about 3.5GB of RAM.

Quantizing a model is as simple as passing a `quantization_config` to the model. Below, we'll leverage the BitsAndyBytes quantization (but refer to [this page](../quantization.md) for other quantization methods):

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

>>> # specify how to quantize the model
>>> quantization_config = BitsAndBytesConfig(
...         load_in_4bit=True,
...         bnb_4bit_quant_type="nf4",
...         bnb_4bit_compute_dtype="torch.float16",
... )

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", quantization_config=True, device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

>>> prompt = "My favourite condiment is"

>>> messages = [
...     {"role": "user", "content": "What is your favourite condiment?"},
...     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
...     {"role": "user", "content": "Do you have mayonnaise recipes?"}
... ]

>>> model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

>>> generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"The expected output"
```

This model was contributed by [Younes Belkada](https://huggingface.co/ybelkada) and [Arthur Zucker](https://huggingface.co/ArthurZ) .
The original code can be found [here](https://github.com/mistralai/mistral-src).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Mistral. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

<PipelineTag pipeline="text-generation"/>

- A demo notebook to perform supervised fine-tuning (SFT) of Mistral-7B can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb). 🌎
- A [blog post](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl) on how to fine-tune LLMs in 2024 using Hugging Face tooling. 🌎
- The [Alignment Handbook](https://github.com/huggingface/alignment-handbook) by Hugging Face includes scripts and recipes to perform supervised fine-tuning (SFT) and direct preference optimization with Mistral-7B. This includes scripts for full fine-tuning, QLoRa on a single GPU as well as multi-GPU fine-tuning.
- [Causal language modeling task guide](../tasks/language_modeling)

## MistralConfig

[[autodoc]] MistralConfig

## MistralModel

[[autodoc]] MistralModel
    - forward

## MistralForCausalLM

[[autodoc]] MistralForCausalLM
    - forward

## MistralForSequenceClassification

[[autodoc]] MistralForSequenceClassification
    - forward

## MistralForTokenClassification

[[autodoc]] MistralForTokenClassification
    - forward

## MistralForQuestionAnswering

[[autodoc]] MistralForQuestionAnswering
- forward

## FlaxMistralModel

[[autodoc]] FlaxMistralModel
    - __call__

## FlaxMistralForCausalLM

[[autodoc]] FlaxMistralForCausalLM
    - __call__

## TFMistralModel

[[autodoc]] TFMistralModel
    - call

## TFMistralForCausalLM

[[autodoc]] TFMistralForCausalLM
    - call

## TFMistralForSequenceClassification

[[autodoc]] TFMistralForSequenceClassification
    - call