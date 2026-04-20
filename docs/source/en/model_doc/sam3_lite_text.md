<!--Copyright 2026 the HuggingFace Team. All rights reserved.

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
*This model was released on 2026-02-12 and added to Hugging Face Transformers on 2026-04-12.*

# SAM3-LiteText

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

## Overview

SAM3-LiteText was proposed in [SAM3-LiteText: An Anatomical Study of the SAM3 Text Encoder for Efficient Vision-Language Segmentation](https://huggingface.co/papers/2602.12173) by Chengxi Zeng, Yuxuan Jiang, Ge Gao, Shuai Wang, Duolikun Danier, Bin Zhu, Stevan Rudinac, David Bull, and Fan Zhang.

SAM3-LiteText is a lightweight variant of [SAM3](sam3) that replaces the heavy SAM3 text encoder (353M parameters) with a compact MobileCLIP-based text encoder optimized through knowledge distillation. The SAM3 ViT-H image encoder is kept intact. This reduces text encoder parameters by up to 88% while maintaining segmentation performance comparable to the original model.

The abstract from the paper is the following:

*Vision-language segmentation models such as SAM3 enable flexible, prompt-driven visual grounding, but inherit large, general-purpose text encoders originally designed for open-ended language understanding. In practice, segmentation prompts are short, structured, and semantically constrained, leading to substantial over-provisioning in text encoder capacity and persistent computational and memory overhead. In this paper, we perform a large-scale anatomical analysis of text prompting in vision-language segmentation, covering 404,796 real prompts across multiple benchmarks. Our analysis reveals severe redundancy: most context windows are underutilized, vocabulary usage is highly sparse, and text embeddings lie on low-dimensional manifold despite high-dimensional representations. Motivated by these findings, we propose SAM3-LiteText, a lightweight text encoding framework that replaces the original SAM3 text encoder with a compact MobileCLIP student that is optimized by knowledge distillation. Extensive experiments on image and video segmentation benchmarks show that SAM3-LiteText reduces text encoder parameters by up to 88%, substantially reducing static memory footprint, while maintaining segmentation performance comparable to the original model.*

The text encoder architecture is based on [MobileCLIP](https://huggingface.co/papers/2311.17049) and comes in three variants:

| Variant | Text Encoder | Text Params | Reduction |
|---|---|---|---|
| SAM3-LiteText-S0-16 | MobileCLIP-S0 | 42.54M | ~88% |
| SAM3-LiteText-S1-16 | MobileCLIP-S1 | 63.53M | ~82% |
| SAM3-LiteText-L-16 | MobileCLIP2-L | 123.80M | ~65% |

This model was contributed by [nielsr](https://huggingface.co/nielsr) and [yonigozlan](https://huggingface.co/yonigozlan).
The original code can be found [here](https://github.com/SimonZeng7108/efficientsam3/tree/sam3_litetext).

## Usage

SAM3-LiteText is a drop-in replacement for SAM3 with a lightweight text encoder. It uses the same processor ([`Sam3Processor`]) and supports the same prompting interface. Refer to the [SAM3 documentation](sam3) for detailed usage examples including text prompts, box prompts, batched inference, and more.

```python
from io import BytesIO

import httpx
from transformers import AutoModel, AutoProcessor
from PIL import Image

model = AutoModel.from_pretrained("yonigozlan/sam3-litetext-s0", device_map="auto")
processor = AutoProcessor.from_pretrained("yonigozlan/sam3-litetext-s0")

image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(BytesIO(httpx.get(image_url).content)).convert("RGB")

inputs = processor(images=image, text="ear", return_tensors="pt").to(model.device)

outputs = model(**inputs)

results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist(),
)[0]

print(f"Found {len(results['masks'])} objects")
```

## Sam3LiteTextConfig

[[autodoc]] Sam3LiteTextConfig

## Sam3LiteTextTextConfig

[[autodoc]] Sam3LiteTextTextConfig

## Sam3LiteTextGeometryEncoderConfig

[[autodoc]] Sam3LiteTextGeometryEncoderConfig

## Sam3LiteTextDETREncoderConfig

[[autodoc]] Sam3LiteTextDETREncoderConfig

## Sam3LiteTextDETRDecoderConfig

[[autodoc]] Sam3LiteTextDETRDecoderConfig

## Sam3LiteTextMaskDecoderConfig

[[autodoc]] Sam3LiteTextMaskDecoderConfig

## Sam3LiteTextTextModel

[[autodoc]] Sam3LiteTextTextModel
    - forward

## Sam3LiteTextModel

[[autodoc]] Sam3LiteTextModel
    - forward

## Sam3LiteTextPreTrainedModel

[[autodoc]] Sam3LiteTextPreTrainedModel
    - forward
