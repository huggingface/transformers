<!--
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the License for the specific
language governing permissions and limitations under the License.
-->

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

# ViTPose

[ViTPose](https://huggingface.co/papers/2204.12484) is a vision transformer-based model for keypoint (pose) estimation. It uses a simple, non-hierarchical [ViT](./vit) backbone and a lightweight decoder head. This architecture simplifies model design, takes advantage of transformer scalability, and can be adapted to different training strategies.

[ViTPose++](https://huggingface.co/papers/2212.04246) improves on ViTPose by incorporating a mixture-of-experts (MoE) module in the backbone and using more diverse pretraining data.

You can find all ViTPose and ViTPose++ checkpoints under the [ViTPose collection](https://huggingface.co/collections/usyd-community/vitpose-677fcfd0a0b2b5c8f79c4335).

---

## Usage

<hfoptions id="usage">
<hfoption id="AutoModel">

```python
import torch
import requests
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForObjectDetection, AutoModelForPoseEstimation

# Step 1: Detect people using a person detector (RT-DETR here)
url = "http://images.cocodataset.org/val2017/000000000139.jpg"
image = Image.open(requests.get(url, stream=True).raw)

person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to("cuda")

inputs = person_image_processor(images=image, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = person_model(**inputs)

results = person_image_processor.post_process_object_detection(
    outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
)

person_boxes = results[0]["boxes"][results[0]["labels"] == 0].cpu().numpy()
person_boxes[:, 2] -= person_boxes[:, 0]
person_boxes[:, 3] -= person_boxes[:, 1]

# Step 2: Run ViTPose on the cropped person(s)
pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
pose_model = AutoModelForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to("cuda")

pose_inputs = pose_processor(image, boxes=[person_boxes], return_tensors="pt").to("cuda")
with torch.no_grad():
    pose_outputs = pose_model(**pose_inputs)

pose_results = pose_processor.post_process_pose_estimation(pose_outputs, boxes=[person_boxes])
```

</hfoption>
</hfoptions>

---

## Quantization Example

```bash
pip install torchao
```

```python
import torch
from torchao.quantization import quantize
from transformers import AutoModelForPoseEstimation

model = AutoModelForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
quantized_model = quantize(model, dtype=torch.qint8)
torch.save(quantized_model.state_dict(), "vitpose-quantized.pt")
```

---

## Keypoint Visualization

<hfoptions id="visualization">
<hfoption id="Supervision">

```python
import supervision as sv

annotator = sv.KeypointAnnotator()
image_with_kpts = annotator.annotate(image=np.array(image), keypoints=pose_results[0]["keypoints"])
Image.fromarray(image_with_kpts)
```

</hfoption>
<hfoption id="OpenCV">

```python
import cv2

np_img = np.array(image).copy()
kpts = pose_results[0]["keypoints"]
for x, y, score in kpts:
    if score > 0.3:
        cv2.circle(np_img, (int(x), int(y)), 3, (0, 255, 0), -1)

cv2.imshow("Pose", np_img)
cv2.waitKey(0)
```

</hfoption>
</hfoptions>

---

## Notes

- ViTPose is a top-down pose estimator, meaning it requires a separate object detector to first locate individuals in the image.
- Use `AutoProcessor` to automatically prepare bounding box and image inputs.
- ViTPose++ supports passing a `dataset_index` parameter to specify which expert head to use.

```python
outputs = pose_model(**pose_inputs, dataset_index=2)  # 0: COCO, 1: AiC, 2: MPII, etc.
```

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

---

## Citation

```bibtex
@article{xu2022vitpose,
  title={ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation},
  author={Xu, Bowen and Zhang, Keyu and Cheng, Xiatian and Lu, Jianfeng and Lu, Jiwen},
  journal={arXiv preprint arXiv:2204.12484},
  year={2022}
}
```
```


