<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

# ViTPose

[ViTPose](https://arxiv.org/abs/2204.12484) is a vision transformer-based model for keypoint (pose) estimation, achieving state-of-the-art performance on MS COCO. It uses a simple, non-hierarchical ViT backbone and lightweight decoder head. Its follow-up, [ViTPose++](https://arxiv.org/abs/2212.04246), incorporates Mixture-of-Experts (MoE) for stronger generalization.

> [!TIP]
> Click on the ViTPose models in the right sidebar for more examples of how to apply ViTPose to pose estimation tasks.

---

## Usage

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pose_estimator = pipeline(
    task="pose-estimation",
    model="usyd-community/vitpose-base-simple"
)

results = pose_estimator("http://images.cocodataset.org/val2017/000000000139.jpg")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from PIL import Image
from transformers import AutoProcessor, VitPoseForPoseEstimation

image = Image.open("path/to/image.jpg")
processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")

inputs = processor(image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
```

</hfoption>
<hfoption id="transformers-cli">

```bash
transformers-cli run --task pose-estimation --model usyd-community/vitpose-base-simple --image path/to/image.jpg
```

</hfoption>
</hfoptions>

---

## Quantization Example

```bash
pip install optimum
```

```python
from optimum.intel import IncQuantizer

quantizer = IncQuantizer.from_pretrained("usyd-community/vitpose-base-simple")
quantizer.quantize(save_directory="./vitpose-quantized")
```

---

## Attention Visualization

```python
from transformers.utils.attention_mask_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer(model)
visualizer.visualize(inputs["attention_mask"])
```

---

## Notes

- ViTPose is a top-down pose estimator: it requires a prior object detector to crop individuals before keypoint prediction.
- ViTPose++ supports passing a `dataset_index` to activate different MoE expert heads.
- Use `AutoProcessor` to handle bounding box preprocessing automatically.
- Ideal for use cases involving MS COCO, WholeBody, MPII, AP-10K, and APT-36K keypoint datasets.

---

## VitPoseImageProcessor

[[autodoc]] VitPoseImageProcessor
  - preprocess
  - post_process_pose_estimation

## VitPoseConfig

[[autodoc]] VitPoseConfig

## VitPoseForPoseEstimation

[[autodoc]] VitPoseForPoseEstimation
  - forward
