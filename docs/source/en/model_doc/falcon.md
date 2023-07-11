<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Falcon

## Overview

Falcon is a state-of-the-art language model trained on the [RefinedWeb dataset](https://arxiv.org/abs/2306.01116). At the time of writing, it is the leading model on the [OpenLLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). 

There is no paper associated with Falcon yet, but for citation information please see [the repository for Falcon-40B](https://huggingface.co/tiiuae/falcon-40b#citation), the highest-performance Falcon model. 

- The model and tokenizer can be loaded via:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b-instruct")
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b-instruct")

inputs = tokenizer("What's the best way to divide a pizza between three people?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
```

- The Falcon tokenizer is a BPE model.

## FalconConfig

[[autodoc]] FalconConfig

## FalconModel

[[autodoc]] FalconModel
    - forward

## FalconForCausalLM

[[autodoc]] FalconForCausalLM
    - forward

## FalconForSequenceClassification

[[autodoc]] FalconForSequenceClassification
    - forward

## FalconForTokenClassification

[[autodoc]] FalconForTokenClassification
    - forward

## FalconForQuestionAnswering

[[autodoc]] FalconForQuestionAnswering
    - forward