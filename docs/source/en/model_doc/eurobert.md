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
*This model was released on 2025-03-07 and added to Hugging Face Transformers on 2026-03-02.*

# EuroBERT

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

## Overview

[EuroBERT](https://huggingface.co/papers/2503.05500) is a multilingual encoder model based on a refreshed transformer architecture, akin to Llama but with bidirectional attention. It supports a mixture of European and widely spoken languages, with sequences of up to 8192 tokens.

You can find all the original EuroBERT checkpoints under the [EuroBERT](https://huggingface.co/collections/EuroBERT/eurobert) collection, or read more about the release in the [EuroBERT blogpost](https://huggingface.co/blog/EuroBERT/release).

The example below demonstrates how to predict the `<|mask|>` token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="EuroBERT/EuroBERT-210m",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create <|mask|> through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "EuroBERT/EuroBERT-210m",
)
model = AutoModelForMaskedLM.from_pretrained(
    "EuroBERT/EuroBERT-210m",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
inputs = tokenizer("Plants create <|mask|> through a process known as photosynthesis.", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"The predicted token is: {predicted_token}")
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plants create <|mask|> through a process known as photosynthesis." | transformers run --task fill-mask --model EuroBERT/EuroBERT-210m --device 0
```

</hfoption>
</hfoptions>

## EuroBertConfig

[[autodoc]] EuroBertConfig

## EuroBertModel

[[autodoc]] EuroBertModel
    - forward

## EuroBertForMaskedLM

[[autodoc]] EuroBertForMaskedLM
    - forward

## EuroBertForSequenceClassification

[[autodoc]] EuroBertForSequenceClassification
    - forward

## EuroBertForTokenClassification

[[autodoc]] EuroBertForTokenClassification
    - forward
