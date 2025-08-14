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

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

# ModernBERT

[ModernBERT](https://huggingface.co/papers/2412.13663) is a modernized version of [`BERT`] trained on 2T tokens. It brings many improvements to the original architecture such as rotary positional embeddings to support sequences of up to 8192 tokens, unpadding to avoid wasting compute on padding tokens, GeGLU layers, and alternating attention.

You can find all the original ModernBERT checkpoints under the [ModernBERT](https://huggingface.co/collections/answerdotai/modernbert-67627ad707a4acbf33c41deb) collection.

> [!TIP]
> Click on the ModernBERT models in the right sidebar for more examples of how to apply ModernBERT to different language tasks.

The example below demonstrates how to predict the `[MASK]` token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="answerdotai/ModernBERT-base",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create [MASK] through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "answerdotai/ModernBERT-base",
)
model = AutoModelForMaskedLM.from_pretrained(
    "answerdotai/ModernBERT-base",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
inputs = tokenizer("Plants create [MASK] through a process known as photosynthesis.", return_tensors="pt").to("cuda")

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
echo -e "Plants create [MASK] through a process known as photosynthesis." | transformers run --task fill-mask --model answerdotai/ModernBERT-base --device 0
```

</hfoption>
</hfoptions>

## ModernBertConfig

[[autodoc]] ModernBertConfig

<frameworkcontent>
<pt>

## ModernBertModel

[[autodoc]] ModernBertModel
    - forward

## ModernBertForMaskedLM

[[autodoc]] ModernBertForMaskedLM
    - forward

## ModernBertForSequenceClassification

[[autodoc]] ModernBertForSequenceClassification
    - forward

## ModernBertForTokenClassification

[[autodoc]] ModernBertForTokenClassification
    - forward

## ModernBertForMultipleChoice

[[autodoc]] ModernBertForMultipleChoice
    - forward

## ModernBertForQuestionAnswering

[[autodoc]] ModernBertForQuestionAnswering
    - forward

### Usage tips

The ModernBert model can be fine-tuned using the HuggingFace Transformers library with its [official script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py) for question-answering tasks.


</pt>
</frameworkcontent>
