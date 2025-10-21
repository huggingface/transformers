<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-12-18 and added to Hugging Face Transformers on 2024-12-19.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# ModernBERT

[ModernBERT](https://huggingface.co/papers/2412.13663) refreshes the traditional encoder architecture by integrating modern improvements such as Rotary Positional Embeddings for handling up to 8192 tokens, Unpadding to optimize processing of mixed-length sequences, GeGLU layers for enhanced performance, Alternating Attention with a sliding window and global attention, Flash Attention for speed, and hardware-co-designed efficiency. Trained on 2 trillion tokens, ModernBERT achieves state-of-the-art results across various classification and retrieval tasks, including code, while being highly efficient in terms of speed and memory usage for inference on common GPUs.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="answerdotai/ModernBERT-base", dtype="auto")
pipeline("Plants create [MASK] through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

inputs = tokenizer("Plants create [MASK] through a process known as photosynthesis.", return_tensors="pt")
outputs = model(**inputs)
mask_token_id = tokenizer.mask_token_id
mask_position = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_word = tokenizer.decode(outputs.logits[0, mask_position].argmax(dim=-1))
print(f"Predicted word: {predicted_word}")
```

</hfoption>
</hfoptions>

## ModernBertConfig

[[autodoc]] ModernBertConfig

## ModernBertModel

[[autodoc]] ModernBertModel
    - forward

## ModernBertForMaskedLM

[[autodoc]] ModernBertForMaskedLM
    - forward

## ModernBertForSequenceClassification

[[autodoc]] ModernBertForSequenceClassification
    - forward

## ModernBertForTokenClassification

[[autodoc]] ModernBertForTokenClassification
    - forward

## ModernBertForMultipleChoice

[[autodoc]] ModernBertForMultipleChoice
    - forward

## ModernBertForQuestionAnswering

[[autodoc]] ModernBertForQuestionAnswering
    - forward

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="fill-mask", model="answerdotai/ModernBERT-base", dtype="auto")
pipeline("The capital of France is [MASK].")
```

