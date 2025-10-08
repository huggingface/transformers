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
*This model was released on 2019-12-11 and added to Hugging Face Transformers on 2020-11-16 and contributed by [formiel](https://huggingface.co/formiel).*

# FlauBERT

[FlauBERT](https://huggingface.co/papers/1912.05372) is a transformer model pretrained using masked language modeling, similar to BERT. Trained on a large and diverse French corpus using the CNRS Jean Zay supercomputer, FlauBERT demonstrates superior performance across various NLP tasks including text classification, paraphrasing, natural language inference, parsing, and word sense disambiguation. The model, along with a unified evaluation protocol called FLUE, is available for further research in French NLP.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline("fill-mask", model="flaubert/flaubert_base_cased", dtype="auto")
pipeline("Les plantes créent [MASK] grâce à un processus appelé photosynthèse.")
```

</hfoption>
<hfoption id="Pipeline">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("flaubert/flaubert_base_cased", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_cased")

inputs = tokenizer("Les plantes créent [MASK] grâce à un processus appelé photosynthèse.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## FlaubertConfig

[[autodoc]] FlaubertConfig

## FlaubertTokenizer

[[autodoc]] FlaubertTokenizer

## FlaubertModel

[[autodoc]] FlaubertModel
    - forward

## FlaubertWithLMHeadModel

[[autodoc]] FlaubertWithLMHeadModel
    - forward

## FlaubertForSequenceClassification

[[autodoc]] FlaubertForSequenceClassification
    - forward

## FlaubertForMultipleChoice

[[autodoc]] FlaubertForMultipleChoice
    - forward

## FlaubertForTokenClassification

[[autodoc]] FlaubertForTokenClassification
    - forward

## FlaubertForQuestionAnsweringSimple

[[autodoc]] FlaubertForQuestionAnsweringSimple
    - forward

## FlaubertForQuestionAnswering

[[autodoc]] FlaubertForQuestionAnswering
    - forward

