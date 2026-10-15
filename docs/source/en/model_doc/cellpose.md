<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-10-15.*

# Cellpose

## Overview

Cellpose-SAM was proposed in [Cellpose-SAM: superhuman generalization for cellular segmentation](https://www.biorxiv.org/content/10.1101/2025.04.28.651001)
by Marius Pachitariu, Michael Rariden and Carsen Stringer. It is a SAM ViT-L image encoder, fine-tuned with 8x8 patches and
global attention in every layer, that predicts per-pixel vertical and horizontal flows and a cell probability map. The
flows are followed to their sinks to group pixels into cell instances.

Images are processed in overlapping 256x256 tiles: [`CellposeImageProcessor`] tiles and normalizes the images, and
[`~CellposeImageProcessor.post_process_instance_segmentation`] stitches the tile predictions and computes the masks.

The original code can be found [here](https://github.com/MouseLand/cellpose).

## Usage example

```python
import torch
from PIL import Image

from transformers import CellposeForInstanceSegmentation, CellposeImageProcessor

model_id = "ArthurZ/cellpose-sam-v2"
processor = CellposeImageProcessor.from_pretrained(model_id)
model = CellposeForInstanceSegmentation.from_pretrained(model_id)

image = Image.open("cells.png")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(pixel_values=inputs.pixel_values)

masks = processor.post_process_instance_segmentation(outputs, target_sizes=inputs.original_sizes)
```

## CellposeConfig

[[autodoc]] CellposeConfig

## CellposeImageProcessor

[[autodoc]] CellposeImageProcessor
    - preprocess
    - post_process_instance_segmentation

## CellposeModel

[[autodoc]] CellposeModel
    - forward

## CellposeForInstanceSegmentation

[[autodoc]] CellposeForInstanceSegmentation
    - forward
