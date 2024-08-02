<!--Copyright 2024 The Qwen Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Qwen2

## Overview

Qwen2 is the new model series of large language models from the Qwen team. Previously, we released the Qwen series, including Qwen2-0.5B, Qwen2-1.5B, Qwen2-7B, Qwen2-57B-A14B, Qwen2-72B, Qwen2-Audio, etc.

### Model Details

Qwen2 is a language model series including decoder language models of different model sizes. For each size, we release the base language model and the aligned chat model. It is based on the Transformer architecture with SwiGLU activation, attention QKV bias, group query attention, mixture of sliding window attention and full attention, etc. Additionally, we have an improved tokenizer adaptive to multiple natural languages and codes.


## Usage tips

`Qwen2-7B` and `Qwen2-7B-Instruct` can be found on the [Huggingface Hub](https://huggingface.co/Qwen)

In the following, we demonstrate how to use `Qwen2-7B-Instruct` for the inference. Note that we have used the ChatML format for dialog, in this demo we show how to leverage `apply_chat_template` for this purpose.

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> device = "cuda" # the device to load the model onto

>>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

>>> prompt = "Give me a short introduction to large language model."

>>> messages = [{"role": "user", "content": prompt}]

>>> text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

>>> model_inputs = tokenizer([text], return_tensors="pt").to(device)

>>> generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)

>>> generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

>>> response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## Qwen2Config

[[autodoc]] Qwen2Config

## Qwen2Tokenizer

[[autodoc]] Qwen2Tokenizer
    - save_vocabulary

## Qwen2TokenizerFast

[[autodoc]] Qwen2TokenizerFast

## Qwen2Model

[[autodoc]] Qwen2Model
    - forward

## Qwen2ForCausalLM

[[autodoc]] Qwen2ForCausalLM
    - forward

## Qwen2ForSequenceClassification

[[autodoc]] Qwen2ForSequenceClassification
    - forward

## Qwen2ForTokenClassification

[[autodoc]] Qwen2ForTokenClassification
    - forward
