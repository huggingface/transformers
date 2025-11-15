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
<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
            <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
            <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
            <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# FG-CLIP2

## Overview

[FG-CLIP2](https://arxiv.org/abs/2510.10921) is a new generation of text-image cross-modal model excels in fine-grained discrimination and embedding. It is the foundation model for fine-grained vision-language understanding in both English and Chinese. 
Across 29 datasets and 8 diverse tasks, it consistently surpasses recent strong baselines such as SigLIP 2 and MetaCLIP 2, achieving the best reported performance to date in both languages. 

You can find all the original FG-CLIP2 checkpoints under the [FG-CLIP2](https://huggingface.co/collections/qihoo360/fg-clip-2-68ecbf9c548623bb78bc7913) collection.


## Usage example

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel


model = AutoModel.from_pretrained("qihoo360/fg-clip2-base", dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained("qihoo360/fg-clip2-base")

url = "https://huggingface.co/spaces/qihoo360/FG-CLIP2-Retrieval-demo/resolve/main/000093.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = [
"一个简约风格的卧室角落，黑色金属衣架上挂着多件米色和白色的衣物，下方架子放着两双浅色鞋子，旁边是一盆绿植，左侧可见一张铺有白色床单和灰色枕头的床。",
"一个简约风格的卧室角落，黑色金属衣架上挂着多件红色和蓝色的衣物，下方架子放着两双黑色高跟鞋，旁边是一盆绿植，左侧可见一张铺有白色床单和灰色枕头的床。",
"一个简约风格的卧室角落，黑色金属衣架上挂着多件米色和白色的衣物，下方架子放着两双运动鞋，旁边是一盆仙人掌，左侧可见一张铺有白色床单和灰色枕头的床。",
"一个繁忙的街头市场，摊位上摆满水果，背景是高楼大厦，人们在喧闹中购物。"
]
# IMPORTANT: we pass `padding=max_length` and `max_length=64` since the model was trained with this
inputs = processor(text=texts, images=image, padding="max_length", max_length=64, return_tensors="pt").to(model.device)

# NOTE Short captions: max_length=64 walk_type="short"(default)
# NOTE Long captions: max_length=196 walk_type="long"

with torch.no_grad():
    outputs = model(**inputs, walk_type="short")

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image)
print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")
```

## Fgclip2Config

[[autodoc]] Fgclip2Config

## Fgclip2TextConfig

[[autodoc]] Fgclip2TextConfig

## Fgclip2VisionConfig

[[autodoc]] Fgclip2VisionConfig

## Fgclip2ImageProcessor

[[autodoc]] Fgclip2ImageProcessor
    - preprocess

## Fgclip2ImageProcessorFast

[[autodoc]] Fgclip2ImageProcessorFast
    - preprocess

## Fgclip2Processor

[[autodoc]] Fgclip2Processor

## Fgclip2Model

[[autodoc]] Fgclip2Model
    - forward
    - get_text_features
    - get_image_features
    - get_image_dense_feature
    - get_image_region_features

## Fgclip2TextModel

[[autodoc]] Fgclip2TextModel
    - forward

## Fgclip2VisionModel

[[autodoc]] Fgclip2VisionModel
    - forward
