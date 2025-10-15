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
*This model was released on 2019-07-26 and added to Hugging Face Transformers on 2020-11-16 and contributed by [julien-c](https://huggingface.co/julien-c).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# RoBERTa

[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://huggingface.co/papers/1907.11692) builds on Google's BERT model by modifying key hyperparameters, including removing the next-sentence pretraining objective and training with larger mini-batches and learning rates. The study highlights the undertraining of BERT and demonstrates that with these adjustments, RoBERTa can match or exceed the performance of subsequent models on benchmarks like GLUE, RACE, and SQuAD. This underscores the significance of certain design choices in language model pretraining.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="FacebookAI/roberta-base", dtype="auto")
pipeline("Plants create <mask> through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("FacebookAI/roberta-base", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

inputs = tokenizer("Plants create <mask> through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## Usage tips

- RoBERTa doesn't have `token_type_ids`. You don't need to indicate which token belongs to which segment.
- Separate segments with the separation token `tokenizer.sep_token` or `</s>`.

## RobertaConfig

[[autodoc]] RobertaConfig

## RobertaTokenizer

[[autodoc]] RobertaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## RobertaTokenizerFast

[[autodoc]] RobertaTokenizerFast
    - build_inputs_with_special_tokens

## RobertaModel

[[autodoc]] RobertaModel
    - forward

## RobertaForCausalLM

[[autodoc]] RobertaForCausalLM
    - forward

## RobertaForMaskedLM

[[autodoc]] RobertaForMaskedLM
    - forward

## RobertaForSequenceClassification

[[autodoc]] RobertaForSequenceClassification
    - forward

## RobertaForMultipleChoice

[[autodoc]] RobertaForMultipleChoice
    - forward

## RobertaForTokenClassification

[[autodoc]] RobertaForTokenClassification
    - forward

## RobertaForQuestionAnswering

[[autodoc]] RobertaForQuestionAnswering
    - forward

