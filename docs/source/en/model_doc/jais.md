<!--Copyright 2024 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# JAIS

## Overview

The JAIS family of models was proposed in [Jais and Jais-chat: Arabic-Centric Foundation and Instruction-Tuned Open Generative Large Language Models](https://arxiv.org/abs/2308.16149) by [Inception](https://www.inceptionai.ai/).

### Model Details

The Jais family of models is a comprehensive series of bilingual English-Arabic large language models (LLMs). These models are optimized to excel in Arabic while having strong English capabilities. All models in this family are auto-regressive language models that use a transformer-based, decoder-only architecture (GPT-3).

Jais models are trained from scratch, incorporating the SwiGLU non-linear activation function and ALiBi position encoding. These architectural enhancements allow the models to extrapolate at long sequence lengths, leading to improved context handling and precision.


| **Pre-trained Model**   | **Fine-tuned Model** | **Size (Parameters)** | **Context length (Tokens)** |
|:---------------------|:--------|:-------|:-------|
| [jais-family-30b-16k](https://huggingface.co/inceptionai/jais-family-30b-16k)   | [Jais-family-30b-16k-chat](https://huggingface.co/inceptionai/jais-family-30b-16k-chat) | 30B | 16,384 |
| [jais-family-30b-8k](https://huggingface.co/inceptionai/jais-family-30b-8k)  | [Jais-family-30b-8k-chat](https://huggingface.co/inceptionai/jais-family-30b-8k-chat) | 30B | 8,192 |
| [jais-family-13b ](https://huggingface.co/inceptionai/jais-family-13b)  | [Jais-family-13b-chat](https://huggingface.co/inceptionai/jais-family-13b-chat) | 13B | 2,048 |
| [jais-family-6p7b](https://huggingface.co/inceptionai/jais-family-6p7b)  | [Jais-family-6p7b-chat](https://huggingface.co/inceptionai/jais-family-6p7b-chat) | 6.7B | 2,048 |
| [jais-family-2p7b](https://huggingface.co/inceptionai/jais-family-2p7b)  | [Jais-family-2p7b-chat](https://huggingface.co/inceptionai/jais-family-2p7b-chat) | 2.7B   | 2,048  |
| [jais-family-1p3b](https://huggingface.co/inceptionai/jais-family-1p3b)  | [Jais-family-1p3b-chat](https://huggingface.co/inceptionai/jais-family-1p3b-chat) | 1.3B | 2,048 |
| [jais-family-590m](https://huggingface.co/inceptionai/jais-family-590m)  | [Jais-family-590m-chat](https://huggingface.co/inceptionai/jais-family-590m-chat)  | 590M   | 2,048  |

The original code for JAIS can be found [here](https://huggingface.co/inceptionai/jais-family-30b-16k).

## Usage tips

All JAIS models ranging from 590m to 70b can be found on the [Huggingface Hub](https://huggingface.co/inceptionai)

In the following, we demonstrate how to use `jais-family-30b-16k-chat` for the inference. Note that JAIS has been integrated in the development version (4.44.1.dev) of `transformers`. Until the official version is released through `pip`, ensure that you are doing one of the following:

* When loading the model, ensure that `trust_remote_code=True` is passed as an argument of the `from_pretrained()` function.

* Update your local `transformers` to the development version: `pip uninstall -y transformers && pip install git+https://github.com/huggingface/transformers`. The previous command is an alternative to cloning and installing from the source.

```python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "inceptionai/jais-family-30b-16k-chat"

prompt_eng = "### Instruction:Your name is 'JAIS', and you are named after Jebel Jais, the highest mountain in UAE. You were made by 'Inception' in the UAE. You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Complete the conversation between [|Human|] and [|AI|]:\n### Input: [|Human|] {Question}\n[|AI|]\n### Response :"
prompt_ar = "### Instruction:اسمك \"جيس\" وسميت على اسم جبل جيس اعلى جبل في الامارات. تم بنائك بواسطة Inception في الإمارات. أنت مساعد مفيد ومحترم وصادق. أجب دائمًا بأكبر قدر ممكن من المساعدة، مع الحفاظ على البقاء أمناً. أكمل المحادثة بين [|Human|] و[|AI|] :\n### Input:[|Human|] {Question}\n[|AI|]\n### Response :"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)


def get_response(text, tokenizer=tokenizer, model=model):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    inputs = input_ids.to(device)
    input_len = inputs.shape[-1]
    generate_ids = model.generate(
        inputs,
        top_p=0.9,
        temperature=0.3,
        max_length=2048,
        min_length=input_len + 4,
        repetition_penalty=1.2,
        do_sample=True,
    )
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    response = response.split("### Response :")[-1]
    return response


ques = "ما هي عاصمة الامارات؟"
text = prompt_ar.format_map({'Question': ques})
print(get_response(text))

ques = "What is the capital of UAE?"
text = prompt_eng.format_map({'Question': ques})
print(get_response(text))

```

## JAISConfig

[[autodoc]] JAISConfig

## JAISModel

[[autodoc]] JAISModel
    - forward

## JAISLMHeadModel

[[autodoc]] JAISLMHeadModel
    - forward

## JAISForSequenceClassification

[[autodoc]] JAISForSequenceClassification
    - forward

## JAISForTokenClassification

[[autodoc]] JAISForTokenClassification
    - forward
