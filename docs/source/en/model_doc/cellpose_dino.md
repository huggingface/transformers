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

# Cellpose-DINO

## Overview

Cellpose-DINO swaps the SAM encoder of [Cellpose](cellpose) for a [DINOv3](dinov3) ViT (ViT-L/16 or ViT-B/16). The
16x16 patch convolution runs with stride 8 (and padding 4), so the model predicts on an 8x8 token grid, and a linear
readout maps each patch token to 8x8 pixels of vertical flow, horizontal flow and cell probability. Images are
processed in 384x384 tiles.

The checkpoints come from the [Cellpose repository](https://github.com/MouseLand/cellpose), see
[Cellpose-SAM: superhuman generalization for cellular segmentation](https://www.biorxiv.org/content/10.1101/2025.04.28.651001).

## Usage example

```python
import torch
from PIL import Image

from transformers import CellposeDinoForInstanceSegmentation, CellposeImageProcessor

model_id = "mouseland/cellpose-dino"
processor = CellposeImageProcessor.from_pretrained(model_id)
model = CellposeDinoForInstanceSegmentation.from_pretrained(model_id)

image = Image.open("cells.png")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(pixel_values=inputs.pixel_values)

masks = processor.post_process_instance_segmentation(outputs, target_sizes=inputs.original_sizes)
```

## CellposeDinoConfig

[[autodoc]] CellposeDinoConfig

## CellposeDinoModel

[[autodoc]] CellposeDinoModel
    - forward

## CellposeDinoForInstanceSegmentation

[[autodoc]] CellposeDinoForInstanceSegmentation
    - forward
