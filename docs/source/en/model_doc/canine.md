<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-03-11 and added to Hugging Face Transformers on 2021-06-30 and contributed by [nielsr](https://huggingface.co/nielsr).*

# CANINE

[CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://huggingface.co/papers/2103.06874) presents CANINE, a neural encoder that processes text directly at the Unicode character level without explicit tokenization or vocabulary. It addresses the challenges of varying language suitability and vocabulary limitations by using a downsampling strategy to manage longer sequences and a deep Transformer stack to capture context. CANINE achieves a 2.8 F1 score improvement on TyDi QA compared to a similar mBERT model, despite having 28% fewer parameters.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-classification", model="google/canine-s", dtype="auto")
pipeline("Plants are amazing because they can create energy from the sun.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("google/canine-s", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("google/canine-s")

inputs = tokenizer("Plants are amazing because they can create energy from the sun.", return_tensors="pt")
outputs = model(**inputs)
predicted_class_id = outputs.logits.argmax(dim=-1).item()
label = model.config.id2label[predicted_class_id]
print(f"Predicted label: {label}")
```

</hfoption>
</hfoptions>

## CanineConfig

[[autodoc]] CanineConfig

## CanineTokenizer

[[autodoc]] CanineTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences

## CANINE specific outputs

[[autodoc]] models.canine.modeling_canine.CanineModelOutputWithPooling

## CanineModel

[[autodoc]] CanineModel
    - forward

## CanineForSequenceClassification

[[autodoc]] CanineForSequenceClassification
    - forward

## CanineForMultipleChoice

[[autodoc]] CanineForMultipleChoice
    - forward

## CanineForTokenClassification

[[autodoc]] CanineForTokenClassification
    - forward

## CanineForQuestionAnswering

[[autodoc]] CanineForQuestionAnswering
    - forward

