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
*This model was contributed to Hugging Face Transformers on 2026-07-28.*


# LoMa

## Overview

LoMa was proposed in [LoMa: Local Feature Matching Revisited](https://huggingface.co/papers/2604.04931) by David Nordström,
Johan Edstedt, Georg Bökman, Jonathan Astermark, Anders Heyden, Viktor Larsson, Mårten Wadenbäck, Michael Felsberg,
and Fredrik Kahl. It is a local feature matcher that refines local descriptors with alternating self- and
cross-attention before selecting mutual matches with a dual-softmax score matrix.

This initial integration supports LoMa with the native SuperPoint keypoint detector. The matching transformer uses
learnable Fourier positional encoding for self-attention and leaves cross-attention position-free, matching the
reference architecture. The local descriptor network is included as an internal component. The conversion utility maps
official LoMa matcher weights (B, L, and G) into the SuperPoint-based model. Since the reference checkpoints pair the
matcher with DaD and DeDoDe, end-to-end numerical parity with the reference pipeline is not expected in this initial
scope.

The original code is available in the [LoMa repository](https://github.com/davnords/LoMa).

## Usage examples

```python
import torch

from transformers import AutoImageProcessor, LoMaConfig, LoMaForKeypointMatching

config = LoMaConfig()
model = LoMaForKeypointMatching(config).eval()
image_processor = AutoImageProcessor.from_config(config)

# `images` is a pair of images, or a batch of image pairs.
inputs = image_processor(images=images, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# `outputs.matches` contains the mutually matched keypoint indices and -1 for unmatched points.
```

## LoMaConfig

[[autodoc]] LoMaConfig

## LoMaPreTrainedModel

[[autodoc]] LoMaPreTrainedModel
    - forward

## LoMaForKeypointMatching

[[autodoc]] LoMaForKeypointMatching

## LoMaImageProcessor

[[autodoc]] LoMaImageProcessor

## LoMaImageProcessorPil

[[autodoc]] LoMaImageProcessorPil
