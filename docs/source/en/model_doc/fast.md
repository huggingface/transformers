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

# FAST

## Overview

Fast model proposes an accurate and efficient scene text detection framework, termed FAST (i.e., faster 
arbitrarily-shaped text detector). 

FAST has two new designs. (1) We design a minimalist kernel representation (only has 1-channel output) to model text 
with arbitrary shape, as well as a GPU-parallel post-processing to efficiently assemble text lines with a negligible 
time overhead. (2) We search the network architecture tailored for text detection, leading to more powerful features 
than most networks that are searched for image classification.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/fast_architecture.png"
alt="drawing" width="600"/>

<small> FAST architecture taken from the <a href="https://arxiv.org/abs/2111.02394">original paper.</a> </small>

This model was contributed by [jadechoghari](https://huggingface.co/jadechoghari), [Raghavan](https://huggingface.co/Raghavan), and [qubvel-hf](https://huggingface.co/qubvel-hf).

## Usage tips 
```py
>>> import torch
>>> import requests

>>> from PIL import Image
>>> from transformers import FastForSceneTextRecognition, FastImageProcessor

>>> image_url = "https://huggingface.co/datasets/Raghavan/fast_model_samples/resolve/main/img657.jpg"
>>> image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

>>> image_processor = FastImageProcessor.from_pretrained("jadechoghari/FAST-tiny-model")
>>> model = FastForSceneTextRecognition.from_pretrained("jadechoghari/FAST-tiny-model")

>>> inputs = image_processor(image, return_tensor="pt")

>>> output = model(pixel_values=torch.tensor(inputs["pixel_values"]))
    target_sizes = [(image.height, image.width)]
    threshold = 0.88
    final_out = image_processor.post_process_text_detection(
        output, target_sizes, threshold, bounding_box_type="rect"
    )
```

## FastConfig

[[autodoc]] FastConfig

## FastImageProcessor

[[autodoc]] FastImageProcessor

## FastForSceneTextRecognition

[[autodoc]] FastForSceneTextRecognition
- forward
