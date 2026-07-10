<!--Copyright 2026 The HuggingFace Team. All rights reserved.
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-07-07.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# NemotronH Omni

NemotronH Omni is a multimodal reasoning model from NVIDIA that pairs the [NemotronH](./nemotron_h) hybrid
Mamba-Transformer language model with a [RADIO](./radio) vision encoder and an optional Parakeet-based sound encoder.
Image (and video) patches are projected through a RADIO tower and a pixel-shuffle MLP into the language model's
embedding space at the `<image>` / `<video>` context-token positions; audio clips are projected in the same way at
`<audio>` positions. The result is a single autoregressive model that reasons jointly over text, images, video and
sound.

The example below demonstrates how to reason over an image and a text prompt with the
[`NemotronH_Omni_Reasoning_V3`] class.

<hfoptions id="usage">
<hfoption id="NemotronH_Omni_Reasoning_V3">

```python
import requests
from PIL import Image

from transformers import (
    NemotronH_Omni_Reasoning_V3,
    NemotronH_Omni_Reasoning_V3ImageProcessor,
    NemotronH_Omni_Reasoning_V3Processor,
    PreTrainedTokenizerFast,
)


model_id = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"

image_processor = NemotronH_Omni_Reasoning_V3ImageProcessor.from_pretrained(model_id)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
processor = NemotronH_Omni_Reasoning_V3Processor(
    image_processor=image_processor, tokenizer=tokenizer, chat_template=tokenizer.chat_template
)
model = NemotronH_Omni_Reasoning_V3.from_pretrained(
    model_id,
    device_map="auto",
).eval()

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
)
accepted = {"input_ids", "attention_mask", "pixel_values", "pixel_values_videos", "sound_clips", "sound_length"}
inputs = {k: v.to(model.device) for k, v in inputs.items() if k in accepted and hasattr(v, "to")}

output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
generated = output[0, inputs["input_ids"].shape[-1] :]
print(processor.tokenizer.decode(generated, skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## NemotronH_Omni_Reasoning_V3_Config

[[autodoc]] NemotronH_Omni_Reasoning_V3_Config

## SoundConfig

[[autodoc]] SoundConfig

## NemotronH_Omni_Reasoning_V3ImageProcessor

[[autodoc]] NemotronH_Omni_Reasoning_V3ImageProcessor

## NemotronH_Omni_Reasoning_V3Processor

[[autodoc]] NemotronH_Omni_Reasoning_V3Processor

## NemotronH_Omni_Reasoning_V3

[[autodoc]] NemotronH_Omni_Reasoning_V3
    - forward
    - generate
