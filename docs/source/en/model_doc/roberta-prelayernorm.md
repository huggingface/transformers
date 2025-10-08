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
*This model was released on 2019-04-01 and added to Hugging Face Transformers on 2022-12-19 and contributed by [andreasmadsen](https://huggingface.co/andreasmadsen).*


# RoBERTa-PreLayerNorm

[RoBERTa-PreLayerNorm](https://huggingface.co/papers/1904.01038) is part of the fairseq toolkit, which facilitates training custom models for tasks like translation and summarization. Built on PyTorch, fairseq supports distributed training, mixed-precision training, and inference on modern GPUs. This specific model variant applies layer normalization before the self-attention and feed-forward layers, differing from the standard RoBERTa configuration.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="andreasmadsen/efficient_mlm_m0.40", dtype="auto")
pipeline("Plants create <mask> through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("andreasmadsen/efficient_mlm_m0.40", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("andreasmadsen/efficient_mlm_m0.40")

inputs = tokenizer("Plants create <mask> through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## RobertaPreLayerNormConfig

[[autodoc]] RobertaPreLayerNormConfig

## RobertaPreLayerNormModel

[[autodoc]] RobertaPreLayerNormModel
    - forward

## RobertaPreLayerNormForCausalLM

[[autodoc]] RobertaPreLayerNormForCausalLM
    - forward

## RobertaPreLayerNormForMaskedLM

[[autodoc]] RobertaPreLayerNormForMaskedLM
    - forward

## RobertaPreLayerNormForSequenceClassification

[[autodoc]] RobertaPreLayerNormForSequenceClassification
    - forward

## RobertaPreLayerNormForMultipleChoice

[[autodoc]] RobertaPreLayerNormForMultipleChoice
    - forward

## RobertaPreLayerNormForTokenClassification

[[autodoc]] RobertaPreLayerNormForTokenClassification
    - forward

## RobertaPreLayerNormForQuestionAnswering

[[autodoc]] RobertaPreLayerNormForQuestionAnswering
    - forward

