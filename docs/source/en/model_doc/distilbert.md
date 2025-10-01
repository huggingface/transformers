<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2019-10-02 and added to Hugging Face Transformers on 2020-11-16.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# DistilBERT

[DistilBERT](https://huggingface.co/papers/1910.01108) is pretrained by knowledge distillation to create a smaller model with faster inference and requires less compute to train. Through a triple loss objective during pretraining, language modeling loss, distillation loss, cosine-distance loss, DistilBERT demonstrates similar performance to a larger transformer language model.

You can find all the original DistilBERT checkpoints under the [DistilBERT](https://huggingface.co/distilbert) organization.

> [!TIP]
> Click on the DistilBERT models in the right sidebar for more examples of how to apply DistilBERT to different language tasks.

The example below demonstrates how to classify text with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">

<hfoption id="Pipeline">

```py
from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    dtype=torch.float16,
    device=0
)

result = classifier("I love using Hugging Face Transformers!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
```

</hfoption>

<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
inputs = tokenizer("I love using Hugging Face Transformers!", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()
predicted_label = model.config.id2label[predicted_class_id]
print(f"Predicted label: {predicted_label}")
```

</hfoption>

<hfoption id="transformers CLI">

```bash
echo -e "I love using Hugging Face Transformers!" | transformers run --task text-classification --model distilbert-base-uncased-finetuned-sst-2-english
```

</hfoption>

</hfoptions>

## Notes

- DistilBERT doesn't have `token_type_ids`, you don't need to indicate which token belongs to which segment. Just
  separate your segments with the separation token `tokenizer.sep_token` (or `[SEP]`).
- DistilBERT doesn't have options to select the input positions (`position_ids` input). This could be added if
  necessary though, just let us know if you need this option.

## DistilBertConfig

[[autodoc]] DistilBertConfig

## DistilBertTokenizer

[[autodoc]] DistilBertTokenizer

## DistilBertTokenizerFast

[[autodoc]] DistilBertTokenizerFast

## DistilBertModel

[[autodoc]] DistilBertModel
    - forward

## DistilBertForMaskedLM

[[autodoc]] DistilBertForMaskedLM
    - forward

## DistilBertForSequenceClassification

[[autodoc]] DistilBertForSequenceClassification
    - forward

## DistilBertForMultipleChoice

[[autodoc]] DistilBertForMultipleChoice
    - forward

## DistilBertForTokenClassification

[[autodoc]] DistilBertForTokenClassification
    - forward

## DistilBertForQuestionAnswering

[[autodoc]] DistilBertForQuestionAnswering
    - forward
