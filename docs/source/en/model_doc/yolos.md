<!--Copyright 2022 The HuggingFace Team. All rights reserved.

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
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# YOLOS

[YOLOS](https://huggingface.co/papers/2106.00666) uses a [Vision Transformer (ViT)](./vit) for object detection with minimal modifications and region priors. It can achieve performance comparable to specialized object detection models and frameworks with knowledge about 2D spatial structures.


You can find all the original YOLOS checkpoints under the [HUST Vision Lab](https://huggingface.co/hustvl/models?search=yolos) organization.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/yolos_architecture.png" alt="drawing" width="600"/>

<small> YOLOS architecture. Taken from the <a href="https://huggingface.co/papers/2106.00666">original paper</a>.</small>


> [!TIP]
> This model wasa contributed by [nielsr](https://huggingface.co/nielsr).
> Click on the YOLOS models in the right sidebar for more examples of how to apply YOLOS to different object detection tasks.

The example below demonstrates how to detect objects with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

detector = pipeline(
    task="object-detection",
    model="hustvl/yolos-base",
    dtype=torch.float16,
    device=0
)
detector("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
```

</hfoption>
<hfoption id="Automodel">

```py
import torch
from PIL import Image
import requests
from transformers import AutoImageProcessor, AutoModelForObjectDetection

processor = AutoImageProcessor.from_pretrained("hustvl/yolos-base")
model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-base", dtype=torch.float16, attn_implementation="sdpa").to("cuda")

url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
inputs = processor(images=image, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits.softmax(-1)
scores, labels = logits[..., :-1].max(-1)
boxes = outputs.pred_boxes

threshold = 0.3
keep = scores[0] > threshold

filtered_scores = scores[0][keep]
filtered_labels = labels[0][keep]
filtered_boxes  = boxes[0][keep]

width, height = image.size
pixel_boxes = filtered_boxes * torch.tensor([width, height, width, height], device=boxes.device)

for score, label, box in zip(filtered_scores, filtered_labels, pixel_boxes):
    x0, y0, x1, y1 = box.tolist()
    print(f"Label {model.config.id2label[label.item()]}: {score:.2f} at [{x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f}]")
```

</hfoption>
</hfoptions>


## Notes
- Use [`YolosImageProcessor`] for preparing images (and optional targets) for the model. Contrary to [DETR](./detr), YOLOS doesn't require a `pixel_mask`.

## Resources

- Refer to these [notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/YOLOS) for inference and fine-tuning with [`YolosForObjectDetection`] on a custom dataset.

## YolosConfig

[[autodoc]] YolosConfig

## YolosImageProcessor

[[autodoc]] YolosImageProcessor
    - preprocess

## YolosImageProcessorFast

[[autodoc]] YolosImageProcessorFast
    - preprocess
    - pad
    - post_process_object_detection

## YolosFeatureExtractor

[[autodoc]] YolosFeatureExtractor
    - __call__
    - pad
    - post_process_object_detection

## YolosModel

[[autodoc]] YolosModel
    - forward

## YolosForObjectDetection

[[autodoc]] YolosForObjectDetection
    - forward
