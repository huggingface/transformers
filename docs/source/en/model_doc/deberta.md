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
*This model was released on 2020-06-05 and added to Hugging Face Transformers on 2020-11-16 and contributed by [DeBERTa](https://huggingface.co/DeBERTa).*

# DeBERTa

[DeBERTa](https://huggingface.co/papers/2006.03654) improves upon BERT and RoBERTa through disentangled attention and an enhanced mask decoder. Disentangled attention uses separate vectors for content and position, computing attention weights with disentangled matrices. The enhanced mask decoder replaces the softmax layer for predicting masked tokens during pretraining. These techniques boost pretraining efficiency and downstream task performance, with DeBERTa outperforming RoBERTa-Large on MNLI, SQuAD v2.0, and RACE using half the training data.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="microsoft/deberta-base", dtype="auto")
pipeline("Plants create [MASK] through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("microsoft/deberta-base", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

inputs = tokenizer("Plants create [MASK] through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## DebertaConfig

[[autodoc]] DebertaConfig

## DebertaTokenizer

[[autodoc]] DebertaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## DebertaTokenizerFast

[[autodoc]] DebertaTokenizerFast
    - build_inputs_with_special_tokens
    - create_token_type_ids_from_sequences

## DebertaModel

[[autodoc]] DebertaModel
    - forward

## DebertaPreTrainedModel

[[autodoc]] DebertaPreTrainedModel

## DebertaForMaskedLM

[[autodoc]] DebertaForMaskedLM
    - forward

## DebertaForSequenceClassification

[[autodoc]] DebertaForSequenceClassification
    - forward

## DebertaForTokenClassification

[[autodoc]] DebertaForTokenClassification
    - forward

## DebertaForQuestionAnswering

[[autodoc]] DebertaForQuestionAnswering
    - forward

