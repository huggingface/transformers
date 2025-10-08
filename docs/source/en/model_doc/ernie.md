<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2019-04-19 and added to Hugging Face Transformers on 2022-09-09 and contributed by [nghuyong](https://huggingface.co/nghuyong).*

# ERNIE

[ERNIE](https://huggingface.co/papers/1904.09223) is a language representation model that extends BERT by introducing knowledge-driven masking strategies. Unlike BERT’s word-level masking, ERNIE applies entity-level masking (covering named entities) and phrase-level masking (covering multi-word concepts) to better capture semantic and contextual information. Experiments show that ERNIE achieves state-of-the-art performance on five major Chinese NLP tasks, including inference, similarity, named entity recognition, sentiment, and question answering. It also demonstrates superior knowledge inference ability in cloze-style evaluations.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="nghuyong/ernie-1.0-base-zh", dtype="auto")
pipeline("植物通过光合作用产生能量。")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("nghuyong/ernie-1.0-base-zh", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")

inputs = tokenizer("植物通过光合作用产生能量。", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, do_sample=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

</hfoption>
</hfoptions>

## ErnieConfig

[[autodoc]] ErnieConfig
    - all

## Ernie specific outputs

[[autodoc]] models.ernie.modeling_ernie.ErnieForPreTrainingOutput

## ErnieModel

[[autodoc]] ErnieModel
    - forward

## ErnieForPreTraining

[[autodoc]] ErnieForPreTraining
    - forward

## ErnieForCausalLM

[[autodoc]] ErnieForCausalLM
    - forward

## ErnieForMaskedLM

[[autodoc]] ErnieForMaskedLM
    - forward

## ErnieForNextSentencePrediction

[[autodoc]] ErnieForNextSentencePrediction
    - forward

## ErnieForSequenceClassification

[[autodoc]] ErnieForSequenceClassification
    - forward

## ErnieForMultipleChoice

[[autodoc]] ErnieForMultipleChoice
    - forward

## ErnieForTokenClassification

[[autodoc]] ErnieForTokenClassification
    - forward

## ErnieForQuestionAnswering

[[autodoc]] ErnieForQuestionAnswering
    - forward