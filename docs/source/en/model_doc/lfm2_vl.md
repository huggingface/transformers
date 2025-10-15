<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-09-18.*

# LFM2-VL

## Overview

[LFM2-VL](https://www.liquid.ai/blog/lfm2-vl-efficient-vision-language-models) is a multimodal model combining a language model backbone (from LFM2-1.2B or LFM2-350M) with a SigLIP2 NaFlex vision encoder and a 2-layer MLP multimodal projector. The vision encoder processes images up to 512×512 resolution natively, handling non-standard aspect ratios and splitting larger images into patches while optionally using thumbnails for global context. The multimodal projector applies pixel unshuffle to reduce image token count, enabling adjustable tradeoffs between speed and quality at inference without retraining. Training involves a staged fusion of language and vision data—shifting from 95% text to 30%—followed by supervised fine-tuning on roughly 100 billion multimodal tokens from diverse open-source and synthetic datasets.

<hfoptions id="usage">
<hfoption i="AutoModelForImageTextToText">

```py
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("LiquidAI/LFM2-VL-1.6B", dtype="uto")
processor = AutoProcessor.from_pretrained("LiquidAI/LFM2-VL-1.6B")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    tokenize=True,
)

outputs = model.generate(**inputs, max_new_tokens=64)
print(processor.batch_decode(outputs, skip_special_tokens=True)[0])
```

</hfoption>
</hfoptions>

## Lfm2VlImageProcessorFast

[[autodoc]] Lfm2VlImageProcessorFast

## Lfm2VlProcessor

[[autodoc]] Lfm2VlProcessor

## Lfm2VlConfig

[[autodoc]] Lfm2VlConfig

## Lfm2VlModel

[[autodoc]] Lfm2VlModel
    - forward

## Lfm2VlForConditionalGeneration

[[autodoc]] Lfm2VlForConditionalGeneration
    - forward
