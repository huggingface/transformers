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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white" >
    </div>
</div>

# RoMa

[RoMa](https://huggingface.co/papers/2305.15404) ("Robust Dense Feature Matching") is a state-of-the-art *dense*
feature matcher. It fuses the robust, coarse features of the frozen [DINOv2](./dinov2) foundation model with a fine
ConvNet feature pyramid to build a precisely localizable feature pyramid. A transformer match decoder with a
Gaussian-Process module predicts anchor probabilities (regression-by-classification) to express the multimodality of
matching, and a cascade of convolutional refiners produces a dense warp field together with a certainty map. Sparse
correspondences are then sampled from the dense prediction.

Because the model produces a dense warp, it is well suited to settings where accuracy matters more than speed, such as
two-view geometry estimation and 3D reconstruction. The same architecture also backs the
[MatchAnything](https://huggingface.co/papers/2501.07556) and [MINIMA](https://github.com/LSXI7/MINIMA) checkpoints,
which improve cross-modal generalization, so they can be loaded with the same modeling code.

> [!TIP]
> Click on the RoMa models in the right sidebar for more examples of how to apply RoMa to different computer vision
> tasks.

The example below demonstrates how to match keypoints between two images with the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoImageProcessor, AutoModelForKeypointMatching
from transformers.image_utils import load_image

image1 = load_image("https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg")
image2 = load_image("https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg")

processor = AutoImageProcessor.from_pretrained("Parskatt/roma_outdoor")
model = AutoModelForKeypointMatching.from_pretrained("Parskatt/roma_outdoor")

inputs = processor([[image1, image2]], return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Sampled sparse correspondences in pixel coordinates of the original images.
target_sizes = [[(image1.height, image1.width), (image2.height, image2.width)]]
matches = processor.post_process_keypoint_matching(outputs, target_sizes, threshold=0.2)
for keypoints in matches:
    print(keypoints["keypoints0"].shape, keypoints["keypoints1"].shape)

# The dense warp and certainty are also available on the output.
print(outputs.warp.shape, outputs.certainty.shape)
```

</hfoption>
</hfoptions>

## Notes

- RoMa is a dense matcher: [`RomaForKeypointMatching`] returns a dense `warp` and `certainty` in addition to the
  sampled sparse `matches` / `matching_scores`. In the default symmetric mode the dense `warp` width is doubled (left
  half: `image0 -> image1`, right half: `image1 -> image0`).
- Input images are resized to a multiple of 14 (560×560 by default) as required by the DINOv2 backbone, and are
  RGB-normalized with the ImageNet statistics.
- The same modeling code loads the official `Parskatt/roma_outdoor` and `roma_indoor` checkpoints as well as the
  cross-modal `MatchAnything`-RoMa and `MINIMA`-RoMa checkpoints (all share the RoMa-v1 architecture).
- **High-resolution refinement.** The pretrained checkpoints refine the coarse warp at a higher resolution by
  default (`config.upsample_predictions=True`, processor `do_upsample=True`). The processor then returns a second,
  higher-resolution `pixel_values_upsampled` tensor, so the example above — `model(**inputs)` — reproduces the
  reference RoMa behaviour and returns the dense outputs at the upsample resolution (864×864 by default). To run only
  the (cheaper) coarse pass, set `config.upsample_predictions=False` or drop `pixel_values_upsampled`; the model
  then returns the 560×560 coarse warp.

## RomaConfig

[[autodoc]] RomaConfig

## RomaImageProcessor

[[autodoc]] RomaImageProcessor
    - preprocess
    - post_process_keypoint_matching

## RomaModel

[[autodoc]] RomaModel
    - forward

## RomaForKeypointMatching

[[autodoc]] RomaForKeypointMatching
    - forward
