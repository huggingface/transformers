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
*This model was released on 2021-01-05 and added to Hugging Face Transformers on 2021-02-26 and contributed by [kssteven](https://huggingface.co/kssteven).*

# I-BERT

[I-BERT](https://huggingface.co/papers/2101.01321) is a quantized version of RoBERTa that performs inference using only integer arithmetic, enabling efficient utilization of integer-only hardware like Turing Tensor Cores and ARM processors. By approximating nonlinear operations such as GELU, Softmax, and Layer Normalization with lightweight integer-only methods, I-BERT achieves end-to-end integer-only inference without floating-point calculations. Evaluations on GLUE tasks with RoBERTa-Base and RoBERTa-Large demonstrate that I-BERT maintains accuracy comparable to full-precision models while achieving a 2.4 to 4.0x speedup for INT8 inference on a T4 GPU.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="kssteven/ibert-roberta-base", dtype="auto")
pipeline("Plants create <mask>> through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("kssteven/ibert-roberta-base", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("kssteven/ibert-roberta-base")

inputs = tokenizer("Plants create <mask>> through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## IBertConfig

[[autodoc]] IBertConfig

## IBertModel

[[autodoc]] IBertModel
    - forward

## IBertForMaskedLM

[[autodoc]] IBertForMaskedLM
    - forward

## IBertForSequenceClassification

[[autodoc]] IBertForSequenceClassification
    - forward

## IBertForMultipleChoice

[[autodoc]] IBertForMultipleChoice
    - forward

## IBertForTokenClassification

[[autodoc]] IBertForTokenClassification
    - forward

## IBertForQuestionAnswering

[[autodoc]] IBertForQuestionAnswering
    - forward

