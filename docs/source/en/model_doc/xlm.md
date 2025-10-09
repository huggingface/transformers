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
*This model was released on 2019-01-22 and added to Hugging Face Transformers on 2020-11-16 and contributed by [thomwolf](https://huggingface.co/thomwolf).*

# XLM

[XLM](https://huggingface.co/papers/1901.07291) extends generative pretraining to multiple languages, demonstrating the effectiveness of cross-lingual pretraining. It proposes two methods for learning XLMs: an unsupervised method using monolingual data and a supervised method leveraging parallel data with a new cross-lingual language model objective. The model achieves state-of-the-art results in cross-lingual classification, unsupervised, and supervised machine translation. Specifically, it improves accuracy by 4.9% on XNLI, BLEU by 34.3 on WMT'16 German-English, and BLEU by 38.5 on WMT'16 Romanian-English, outperforming previous best approaches.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="FacebookAI/xlm-roberta-base", dtype="auto")
pipeline("Les plantes créent <mask> grâce à un processus appelé photosynthèse.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("FacebookAI/xlm-roberta-base", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

inputs = tokenizer("Les plantes créent <mask> grâce à un processus appelé photosynthèse.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## XLMConfig

[[autodoc]] XLMConfig

## XLMTokenizer

[[autodoc]] XLMTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## XLM specific outputs

[[autodoc]] models.xlm.modeling_xlm.XLMForQuestionAnsweringOutput

## XLMModel

[[autodoc]] XLMModel
    - forward

## XLMWithLMHeadModel

[[autodoc]] XLMWithLMHeadModel
    - forward

## XLMForSequenceClassification

[[autodoc]] XLMForSequenceClassification
    - forward

## XLMForMultipleChoice

[[autodoc]] XLMForMultipleChoice
    - forward

## XLMForTokenClassification

[[autodoc]] XLMForTokenClassification
    - forward

## XLMForQuestionAnsweringSimple

[[autodoc]] XLMForQuestionAnsweringSimple
    - forward

## XLMForQuestionAnswering

[[autodoc]] XLMForQuestionAnswering
    - forward

