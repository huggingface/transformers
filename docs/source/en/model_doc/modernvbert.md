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
*This model was released on 2025-10-01 and added to Hugging Face Transformers on 2026-01-26.*

# ModernVBert

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

ModernVBert is a Vision-Language encoder that combines [ModernBert](modernbert) with a [SigLIP](siglip) vision encoder. It is optimized for visual document understanding and retrieval tasks.

The model was introduced in [ModernVBERT: Towards Smaller Visual Document Retrievers](https://huggingface.co/papers/2510.01149).

<hfoptions id="usage">
<hfoption id="Python">

```python
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoModelForMaskedLM, AutoProcessor

processor = AutoProcessor.from_pretrained("./mvb")
model = AutoModelForMaskedLM.from_pretrained("./mvb")

image = Image.open(hf_hub_download("HuggingFaceTB/SmolVLM", "example_images/rococo.jpg", repo_type="space"))
text = "This [MASK] is on the wall."

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": text}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages)
inputs = processor(text=prompt, images=[image], return_tensors="pt")

# Inference
with torch.no_grad():
  outputs = model(**inputs)

# To get predictions for the mask:
masked_index = inputs["input_ids"][0].tolist().index(processor.tokenizer.mask_token_id)
predicted_token_id = outputs.logits[0, masked_index].argmax(axis=-1)
predicted_token = processor.tokenizer.decode(predicted_token_id)
print("Predicted token:", predicted_token)  # Predicted token: painting
```

</hfoption>
</hfoptions>

## ModernVBertConfig

[[autodoc]] ModernVBertConfig

## ModernVBertImageProcessor

[[autodoc]] ModernVBertImageProcessor
    - preprocess

## ModernVBertImageProcessorFast

[[autodoc]] ModernVBertImageProcessorFast
    - preprocess

## ModernVBertProcessor

[[autodoc]] ModernVBertProcessor

## ModernVBertModel

[[autodoc]] ModernVBertModel
    - forward

## ModernVBertForMaskedLM

[[autodoc]] ModernVBertForMaskedLM
    - forward

## ModernVBertForSequenceClassification

[[autodoc]] ModernVBertForSequenceClassification
    - forward

## ModernVBertForTokenClassification

[[autodoc]] ModernVBertForTokenClassification
    - forward

## ModernVBertForQuestionAnswering

[[autodoc]] ModernVBertForQuestionAnswering
    - forward

## ModernVBertForMultipleChoice

[[autodoc]] ModernVBertForMultipleChoice
    - forward

