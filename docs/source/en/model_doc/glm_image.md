<!--Copyright 2025 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->


# GlmImage

## Overview

The GlmImage model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.
<INSERT SHORT SUMMARY HERE>

The abstract from the paper is the following:

<INSERT PAPER ABSTRACT HERE>

Tips:

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [INSERT YOUR HF USERNAME HERE](https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).
The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).

## Usage examples

Using GLM-Image with image input to generate vision token for DIT using.

```python
from transformers import AutoProcessor, GlmImageForConditionalGeneration
from accelerate import Accelerator
import torch

device = Accelerator().device

processor = AutoProcessor.from_pretrained("zai-org/GLM-Image")
model = GlmImageForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path="zai-org/GLM-Image",
    dtype=torch.bfloat16,
    device_map=device
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "test.jpg",
            },
            {
                "type": "text",
                "text": "一幅充满可爱漫画风格的文字设计作品，主体为“Taro”字样，采用洁净明亮的纯白色圆润字体，字形饱满柔和，略带手写漫画感，背景以温柔细腻的芋泥紫作为底色，呈现出柔雾般的渐变效果，周围点缀着小星星、心形与气泡等轻盈卡通元素，整体氛围轻快甜美，光线柔和如午后阳光，从左上方洒下微暖的光晕，为画面增添立体感与温馨感，适合呈现梦幻、治愈的视觉体验。<sop>36 24<eop>",
            },
        ],
    }
]
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True).to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=True)
output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
print(output_text)
```

## GlmImageConfig

[[autodoc]] GlmImageConfig

## GlmImageVisionConfig

[[autodoc]] GlmImageVisionConfig

## GlmImageTextConfig

[[autodoc]] GlmImageTextConfig

## GlmImageVQVAEConfig

[[autodoc]] GlmImageVQVAEConfig

## GlmImageImageProcessor

[[autodoc]] GlmImageImageProcessor
    - preprocess

## GlmImageProcessor

[[autodoc]] GlmImageProcessor

## GlmImageVisionModel

[[autodoc]] GlmImageVisionModel
    - forward

## GlmImageTextModel

[[autodoc]] GlmImageTextModel
    - forward

## GlmImageVQVAE

[[autodoc]] GlmImageVQVAE
    - forward

## GlmImageModel

[[autodoc]] GlmImageModel
    - forward

## GlmImageForConditionalGeneration

[[autodoc]] GlmImageForConditionalGeneration
    - forward
