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

You can find the converted checkpoints below. They hold the original weights untouched and are converted on the fly at
load time.

| Checkpoint | Encoder | Tile size |
|---|---|---|
| [ArthurZ/cellpose-sam-v2](https://huggingface.co/ArthurZ/cellpose-sam-v2) | SAM ViT-L, 8x8 patches | 256 |
| [ArthurZ/cellpose-sam](https://huggingface.co/ArthurZ/cellpose-sam) | SAM ViT-L, 8x8 patches | 256 |
| [ArthurZ/cellpose-dino](https://huggingface.co/ArthurZ/cellpose-dino) | [DINOv3](cellpose_dino) ViT-L/16 | 384 |
| [ArthurZ/cellpose-dino-vitb](https://huggingface.co/ArthurZ/cellpose-dino-vitb) | [DINOv3](cellpose_dino) ViT-B/16 | 384 |

The example below shows how to segment cells with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

segmenter = pipeline("image-segmentation", model="ArthurZ/cellpose-sam-v2", device=0)
cells = segmenter("https://www.cellpose.org/static/images/img02.png")
print(len(cells))  # 99, one {"score", "label", "mask"} dict per cell
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoImageProcessor, AutoModelForInstanceSegmentation
from transformers.image_utils import load_image

model_id = "ArthurZ/cellpose-sam-v2"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForInstanceSegmentation.from_pretrained(model_id, device_map="auto")

image = load_image("https://www.cellpose.org/static/images/img02.png")
inputs = processor(image, return_tensors="pt").to(model.device)  # (num_tiles, 3, 256, 256)
with torch.no_grad():
    outputs = model(**inputs)

result = processor.post_process_instance_segmentation(outputs, target_sizes=[(image.height, image.width)])[0]
masks = result["segmentation"]  # (height, width), 0 = background, cells labelled 1..N
print(masks.max())  # 99
```

</hfoption>
</hfoptions>

### Visualizing the masks

`post_process_instance_segmentation` also returns the stitched `flows` (`(2, height, width)`) and the
`cell_probability` logits, which are useful to inspect what the model sees.

```python
import matplotlib.pyplot as plt
import numpy as np

flow_angle = torch.atan2(result["flows"][0], result["flows"][1]).cpu()
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(image)
axes[1].imshow(np.ma.masked_equal(masks.cpu().numpy(), 0) % 20, cmap="tab20", interpolation="nearest")
axes[2].imshow(flow_angle, cmap="hsv")
axes[3].imshow(result["cell_probability"].sigmoid().cpu())
for ax, title in zip(axes, ["image", "masks", "flow direction", "cell probability"]):
    ax.set_title(title)
    ax.axis("off")
plt.show()
```

### Input images

- Images of any size are accepted: they are normalized per channel (1st-99th percentile), padded and cut into
  overlapping tiles of `tile_size`. Pass the original `(height, width)` of every image as `target_sizes` to stitch the
  tiles back.
- Grayscale, 2-channel (for example cytoplasm + nuclei) and RGB images all work. The channel axis is guessed like in
  Cellpose (the smallest dimension); pass `input_data_format="channels_first"` or `"channels_last"` to set it. Only
  the first 3 channels are used.
- Several images can be processed at once. Their tiles are concatenated along the batch dimension, in order:

```python
images = [load_image(url) for url in urls]
inputs = processor(images, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)
results = processor.post_process_instance_segmentation(outputs, target_sizes=[(im.height, im.width) for im in images])
```

### Tuning the masks

| Argument of `post_process_instance_segmentation` | Default | Effect |
|---|---|---|
| `cellprob_threshold` | `0.0` | Lower it to find more (dimmer) cells, raise it to find fewer. |
| `flow_threshold` | `0.4` | Maximum flow error of a mask. Raise it to keep more (irregular) masks, `0` disables the check. |
| `min_size` | `15` | Masks with fewer pixels are removed. |
| `max_size_fraction` | `0.4` | Masks covering a larger fraction of the image are removed. |
| `num_iterations` | `200` | Number of steps used to follow the flows. |

Cellpose-SAM is trained to be robust to cell size, so there is no `diameter` argument. Rescale the image yourself if
cells are much smaller than 10 pixels or larger than 100 pixels in diameter.

### Speed

The model runs on all the tiles of all the images in a single forward pass. Load it in `bfloat16` and compile it to
segment large batches faster:

```python
model = AutoModelForInstanceSegmentation.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
model = torch.compile(model)
```

Post-processing runs on the device of `outputs`, keep it on GPU for large images.

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
