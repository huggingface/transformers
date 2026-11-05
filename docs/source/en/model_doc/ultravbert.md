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
*This model was contributed to Hugging Face Transformers on 2026-10-20.*

# UltraVBert

<div class="flex flex-wrap space-x-1">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

UltraVBert extends [ModernVBert](./modernvbert.md) by pairing the [UltraBERT](./ultrabert.md) text encoder with a [SigLIP](./siglip.md) vision encoder for visual document understanding and retrieval. The image processor splits a page into 512-pixel tiles plus a global view. The connector merges each 4 x 4 block of vision patches into one token, so a tile becomes 64 image tokens, and RMS-normalizes them to the scale of UltraBERT's token embeddings. `max_image_tiles`, 27 by default, caps the tiles per page by lowering the resize edge in 512-pixel steps.

## Usage example

<hfoptions id="usage">
<hfoption id="AutoModel">

```python
import torch

from transformers import AutoModelForMaskedLM, AutoProcessor
from transformers.image_utils import load_image


processor = AutoProcessor.from_pretrained("athrael-soju/UltraVBERT")
model = AutoModelForMaskedLM.from_pretrained("athrael-soju/UltraVBERT", device_map="auto")

image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/receipt_00008.png"
)
inputs = processor(text="<image>Rounding 0 <|mask|> 50,000", images=[image], return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits

masked_index = torch.where(inputs["input_ids"] == processor.tokenizer.mask_token_id)
predicted_token_id = logits[masked_index].argmax(dim=-1)
print(processor.tokenizer.decode(predicted_token_id))
```

</hfoption>
<hfoption id="Pipeline">

The `fill-mask` pipeline reads text only.

```python
from transformers import pipeline


fill_mask = pipeline(task="fill-mask", model="athrael-soju/UltraVBERT", device_map="auto")
print(fill_mask("Plants create <|mask|> through a process known as photosynthesis."))
```

</hfoption>
</hfoptions>

## UltraVBertConfig

[[autodoc]] UltraVBertConfig

## UltraVBertImageProcessor

[[autodoc]] UltraVBertImageProcessor
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
