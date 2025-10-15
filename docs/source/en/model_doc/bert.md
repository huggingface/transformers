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
*This model was released on 2018-10-11 and added to Hugging Face Transformers on 2020-11-16 and contributed by [thomwolf](https://huggingface.co/thomwolf).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# BERT

[BERT](https://huggingface.co/papers/1810.04805) introduces a bidirectional transformer model for language representation, pre-trained using masked language modeling and next sentence prediction. BERT achieves state-of-the-art results across various NLP tasks by fine-tuning with minimal task-specific modifications, significantly improving benchmarks like GLUE, MultiNLI, and SQuAD.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="google-bert/bert-base-uncased", dtype="auto")
pipeline("Plants create [MASK] through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

inputs = tokenizer("Plants create [MASK] through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## Usage tips

- Pad inputs on the right. BERT uses absolute position embeddings.

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

## Bert specific outputs

[[autodoc]] models.bert.modeling_bert.BertForPreTrainingOutput

] models.bert.modeling_flax_bert.FlaxBertForPreTrainingOutput

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
