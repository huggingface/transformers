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

Cellpose-DINO shares the [`CellposeImageProcessor`] of [Cellpose](cellpose), see its page for visualization, batching
and tuning. The checkpoints are [ArthurZ/cellpose-dino](https://huggingface.co/ArthurZ/cellpose-dino) (ViT-L) and
[ArthurZ/cellpose-dino-vitb](https://huggingface.co/ArthurZ/cellpose-dino-vitb) (ViT-B).

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

segmenter = pipeline("image-segmentation", model="ArthurZ/cellpose-dino", device=0)
cells = segmenter("https://www.cellpose.org/static/images/img02.png")
print(len(cells))  # 100
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoImageProcessor, AutoModelForInstanceSegmentation
from transformers.image_utils import load_image

model_id = "ArthurZ/cellpose-dino"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForInstanceSegmentation.from_pretrained(model_id, device_map="auto")

image = load_image("https://www.cellpose.org/static/images/img02.png")
inputs = processor(image, return_tensors="pt").to(model.device)  # (num_tiles, 3, 384, 384)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.post_process_instance_segmentation(outputs, target_sizes=[(image.height, image.width)])[0]["segmentation"]
```

</hfoption>
</hfoptions>

## CellposeDinoConfig

[[autodoc]] CellposeDinoConfig

## CellposeDinoModel

[[autodoc]] CellposeDinoModel
    - forward

## CellposeDinoForInstanceSegmentation

[[autodoc]] CellposeDinoForInstanceSegmentation
    - forward
