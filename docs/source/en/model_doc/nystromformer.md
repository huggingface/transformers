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
*This model was released on 2021-02-07 and added to Hugging Face Transformers on 2022-01-11 and contributed by [novice03](https://huggingface.co/novice03).*

# Nyströmformer

[Nyströmformer](https://huggingface.co/papers/2102.03902) addresses the quadratic complexity issue of self-attention in Transformers by adapting the Nyström method to approximate standard self-attention with O(n) complexity. This enables the model to handle longer sequences containing thousands of tokens. Evaluations on GLUE benchmark and IMDB reviews show that Nyströmformer performs comparably or better than standard self-attention. On longer sequence tasks in the Long Range Arena (LRA) benchmark, it outperforms other efficient self-attention methods.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="uw-madison/nystromformer-512", dtype="auto")
pipeline("Plants create <mask> through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("uw-madison/nystromformer-512", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("uw-madison/nystromformer-512")

inputs = tokenizer("Plants create <mask> through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## NystromformerConfig

[[autodoc]] NystromformerConfig

## NystromformerModel

[[autodoc]] NystromformerModel
    - forward

## NystromformerForMaskedLM

[[autodoc]] NystromformerForMaskedLM
    - forward

## NystromformerForSequenceClassification

[[autodoc]] NystromformerForSequenceClassification
    - forward

## NystromformerForMultipleChoice

[[autodoc]] NystromformerForMultipleChoice
    - forward

## NystromformerForTokenClassification

[[autodoc]] NystromformerForTokenClassification
    - forward

## NystromformerForQuestionAnswering

[[autodoc]] NystromformerForQuestionAnswering
    - forward

