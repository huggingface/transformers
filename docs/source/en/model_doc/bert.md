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
*This model was released on 2018-10-11 and added to Hugging Face Transformers on 2020-11-16.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# BERT

[BERT](https://huggingface.co/papers/1810.04805) is a bidirectional transformer pretrained on unlabeled text to predict masked tokens in a sentence and to predict whether one sentence follows another. The main idea is that by randomly masking some tokens, the model can train on text to the left and right, giving it a more thorough understanding. BERT is also very versatile because its learned language representations can be adapted for other NLP tasks by fine-tuning an additional layer or head.

You can find all the original BERT checkpoints under the [BERT](https://huggingface.co/collections/google/bert-release-64ff5e7a4be99045d1896dbc) collection.

> [!TIP]
> Click on the BERT models in the right sidebar for more examples of how to apply BERT to different language tasks.

The example below demonstrates how to predict the `[MASK]` token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="google-bert/bert-base-uncased",
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
    "google-bert/bert-base-uncased",
)
model = AutoModelForMaskedLM.from_pretrained(
    "google-bert/bert-base-uncased",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
inputs = tokenizer("Plants create [MASK] through a process known as photosynthesis.", return_tensors="pt").to(model.device)

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
echo -e "Plants create [MASK] through a process known as photosynthesis." | transformers run --task fill-mask --model google-bert/bert-base-uncased --device 0
```

</hfoption>
</hfoptions>

## Notes

- Inputs should be padded on the right because BERT uses absolute position embeddings.

## BertConfig

[[autodoc]] BertConfig
    - all

## BertTokenizer

[[autodoc]] BertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## BertTokenizerFast

[[autodoc]] BertTokenizerFast

## BertModel

[[autodoc]] BertModel
    - forward

## BertForPreTraining

[[autodoc]] BertForPreTraining
    - forward

## BertLMHeadModel

[[autodoc]] BertLMHeadModel
    - forward

## BertForMaskedLM

[[autodoc]] BertForMaskedLM
    - forward

## BertForNextSentencePrediction

[[autodoc]] BertForNextSentencePrediction
    - forward

## BertForSequenceClassification

[[autodoc]] BertForSequenceClassification
    - forward

## BertForMultipleChoice

[[autodoc]] BertForMultipleChoice
    - forward

## BertForTokenClassification

[[autodoc]] BertForTokenClassification
    - forward

## BertForQuestionAnswering

[[autodoc]] BertForQuestionAnswering
    - forward

## Bert specific outputs

[[autodoc]] models.bert.modeling_bert.BertForPreTrainingOutput
