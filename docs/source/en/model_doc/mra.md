<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2022-07-21 and added to Hugging Face Transformers on 2023-07-10 and contributed by [novice03](https://huggingface.co/novice03).*

# MRA

[MRA](https://huggingface.co/papers/2207.10284) revisits classical Multiresolution Analysis (MRA) concepts, particularly Wavelets, to approximate the self-attention matrix in Transformers. By leveraging empirical feedback and design choices that consider modern hardware and implementation challenges, the MRA-based approach achieves excellent performance across various criteria. Experiments show that this multi-resolution scheme outperforms most efficient self-attention methods and is effective for both short and long sequences.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="uw-madison/mra-base-512-4", dtype="auto")
pipeline("Plants create <mask> through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("uw-madison/mra-base-512-4", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("uw-madison/mra-base-512-4")

inputs = tokenizer("Plants create <mask> through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## MraConfig

[[autodoc]] MraConfig

## MraModel

[[autodoc]] MraModel
    - forward

## MraForMaskedLM

[[autodoc]] MraForMaskedLM
    - forward

## MraForSequenceClassification

[[autodoc]] MraForSequenceClassification
    - forward

## MraForMultipleChoice

[[autodoc]] MraForMultipleChoice
    - forward

## MraForTokenClassification

[[autodoc]] MraForTokenClassification
    - forward

## MraForQuestionAnswering

[[autodoc]] MraForQuestionAnswering
    - forward

