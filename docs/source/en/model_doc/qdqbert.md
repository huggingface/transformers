<!--Copyright 2021 NVIDIA Corporation and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2020-04-20 and added to Hugging Face Transformers on 2023-06-20 and contributed by [shangz](https://huggingface.co/shangz).*

# QDQBERT

[QDQBERT](https://huggingface.co/papers/2004.09602) explores integer quantization to decrease Deep Neural Network sizes and enhance inference speed through high-throughput integer instructions. The paper examines quantization parameters and evaluates their impact across various neural network models in vision, speech, and language domains. It highlights techniques compatible with processors featuring high-throughput integer pipelines. A workflow for 8-bit quantization is introduced, ensuring accuracy within 1% of the floating-point baseline across all studied networks, including challenging models like MobileNets and BERT-large.

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

## QDQBertConfig

[[autodoc]] QDQBertConfig

## QDQBertModel

[[autodoc]] QDQBertModel
    - forward

## QDQBertLMHeadModel

[[autodoc]] QDQBertLMHeadModel
    - forward

## QDQBertForMaskedLM

[[autodoc]] QDQBertForMaskedLM
    - forward

## QDQBertForSequenceClassification

[[autodoc]] QDQBertForSequenceClassification
    - forward

## QDQBertForNextSentencePrediction

[[autodoc]] QDQBertForNextSentencePrediction
    - forward

## QDQBertForMultipleChoice

[[autodoc]] QDQBertForMultipleChoice
    - forward

## QDQBertForTokenClassification

[[autodoc]] QDQBertForTokenClassification
    - forward

## QDQBertForQuestionAnswering

[[autodoc]] QDQBertForQuestionAnswering
    - forward

