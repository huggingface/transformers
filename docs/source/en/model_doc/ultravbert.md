<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# UltraVBert

<div class="flex flex-wrap space-x-1">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

UltraVBert pairs the [UltraBERT](ultrabert) text encoder with a [SigLIP](siglip) vision encoder for visual document understanding and retrieval, following [ModernVBert](modernvbert). The image processor splits a page into 512-pixel tiles plus a global view. The connector merges each 4 x 4 block of vision patches into one token, so a tile becomes 64 image tokens, and RMS-normalizes them to the scale of UltraBERT's token embeddings. `max_image_tiles` caps the tiles per page by lowering the resize edge in 512-pixel steps.

## Usage example

```python
import torch
from PIL import Image
from transformers import AutoModelForMaskedLM, AutoProcessor

processor = AutoProcessor.from_pretrained("athrael-soju/UltraVBERT")
model = AutoModelForMaskedLM.from_pretrained("athrael-soju/UltraVBERT", device_map="auto")

page = Image.open("page.png")
inputs = processor(text="<image>The table reports revenue in <|mask|>.", images=[page], return_tensors="pt").to(model.device)
with torch.no_grad():
    logits = model(**inputs).logits

masked_logits = logits[inputs.input_ids == processor.tokenizer.mask_token_id]
print(processor.tokenizer.batch_decode(masked_logits.argmax(dim=-1)[:, None]))
```

## UltraVBertConfig

[[autodoc]] UltraVBertConfig

## UltraVBertImageProcessor

[[autodoc]] UltraVBertImageProcessor
    - preprocess

## UltraVBertImageProcessorPil

[[autodoc]] UltraVBertImageProcessorPil
    - preprocess

## UltraVBertModel

[[autodoc]] UltraVBertModel
    - forward

## UltraVBertForMaskedLM

[[autodoc]] UltraVBertForMaskedLM
    - forward

## UltraVBertForSequenceClassification

[[autodoc]] UltraVBertForSequenceClassification
    - forward

## UltraVBertForTokenClassification

[[autodoc]] UltraVBertForTokenClassification
    - forward
