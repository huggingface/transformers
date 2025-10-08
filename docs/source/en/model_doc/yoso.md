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
*This model was released on 2021-11-18 and added to Hugging Face Transformers on 2022-01-26 and contributed by [novice03](https://huggingface.co/novice03).*

# YOSO

[YOSO](https://huggingface.co/papers/2111.09714) approximates standard softmax self-attention using a Bernoulli sampling scheme based on Locality Sensitive Hashing (LSH). This method reduces the quadratic complexity of self-attention to linear by sampling Bernoulli random variables with a single hash. The approach is modified for GPU deployment and evaluated on the GLUE and Long Range Arena (LRA) benchmarks, showing favorable performance and efficiency improvements over standard Transformers.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="uw-madison/yoso-4096", dtype="auto")
pipeline("Plants create <mask> through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("uw-madison/yoso-4096", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("uw-madison/yoso-4096")

inputs = tokenizer("Plants create <mask> through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## YosoConfig

[[autodoc]] YosoConfig

## YosoModel

[[autodoc]] YosoModel
    - forward

## YosoForMaskedLM

[[autodoc]] YosoForMaskedLM
    - forward

## YosoForSequenceClassification

[[autodoc]] YosoForSequenceClassification
    - forward

## YosoForMultipleChoice

[[autodoc]] YosoForMultipleChoice
    - forward

## YosoForTokenClassification

[[autodoc]] YosoForTokenClassification
    - forward

## YosoForQuestionAnswering

[[autodoc]] YosoForQuestionAnswering
    - forward

