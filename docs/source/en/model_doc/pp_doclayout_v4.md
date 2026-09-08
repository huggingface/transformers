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
*This model was contributed to Hugging Face Transformers on 2026-09-08.*

# PP-DocLayoutV4

## Overview

**PP-DocLayoutV4** is an end-to-end document layout analysis model that predicts, in a single forward pass, what each
layout element is, the quadrilateral that encloses it, and the logits the reading order is decoded from.

## Model Architecture

PP-DocLayoutV4 keeps the RT-DETR skeleton of its predecessor — an HGNetV2 backbone, an AIFI hybrid encoder and a
deformable decoder — and changes three things:

- **Quadrilaterals instead of masks.** [`PPDocLayoutV3ForObjectDetection`] predicted a segmentation mask per query and
  fitted a polygon to it. PP-DocLayoutV4 drops the mask branch entirely and directly regresses a four point quad,
  encoded as `[center_x, center_y, dx1, dy1, ..., dx4, dy4]` in sigmoid space. This handles skewed and perspective
  distorted documents without paying for a mask head, and the stride 4 backbone feature is no longer needed.
- **Untied per-layer heads.** Every decoder layer owns its own box and classification head rather than sharing a
  single head with the encoder.
- **Two reading order heads plus a fusion step.** Alongside the relative order head from V3 (antisymmetric: "is `i`
  read before `j`?") there is a successor head (ROOR: "does `j` directly follow `i`?"). [`PPDocLayoutV4S2RFusion`]
  folds a damped transitive closure of the successor matrix back into the relative order logits.

The reading order is not decoded inside the model. `PPDocLayoutV4ForObjectDetection` returns the raw
`relative_order_logits` and `successor_order_logits`, and
[`~PPDocLayoutV4ImageProcessor.post_process_object_detection`] turns them into ranks by thresholding the successor
matrix into a directed acyclic graph (DAG), breaking cycles, sorting each connected component topologically, and
ordering the components by their relative order votes.

## Usage

Results returned by [`~PPDocLayoutV4ImageProcessor.post_process_object_detection`] are already sorted by reading
order. Each result carries `polygon_points` — a `(num_boxes, 4, 2)` tensor of the regressed corners in top-left,
top-right, bottom-right, bottom-left order — plus `boxes`, the axis aligned rectangle enclosing each quad, and
`order_seq`, the 0-based rank of each box. The [`AutoModel`] example below prints `order_seq` directly, the others
print the 1-based position in the sorted list, which is `order_seq + 1` unless one query was kept under several
labels.

<hfoptions id="usage">
<hfoption id="Pipeline">

[`Pipeline`] preserves the reading order in the order of the returned list, but flattens each detection to a score, a
label and an integer bounding box. Use the [`AutoModel`] path below to get the quadrilaterals and `order_seq`.

```python
from transformers import pipeline
from transformers.image_utils import load_image


image = load_image("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg")
layout_detector = pipeline("object-detection", model="PaddlePaddle/PP-DocLayoutV4_safetensors")
results = layout_detector(image)
for idx, res in enumerate(results):
    print(f"Order {idx + 1}: {res}")
```

</hfoption>

<hfoption id="AutoModel">

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers.image_utils import load_image


model_path = "PaddlePaddle/PP-DocLayoutV4_safetensors"
model = AutoModelForObjectDetection.from_pretrained(model_path, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = load_image("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg")
inputs = image_processor(images=image, return_tensors="pt").to(model.device)

outputs = model(**inputs)
results = image_processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]])
for result in results:
    for score, label_id, box, quad, rank in zip(
        result["scores"], result["labels"], result["boxes"], result["polygon_points"], result["order_seq"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Order {rank.item()}: {model.config.id2label[label_id.item()]}, score: {score.item():.2f}, box: {box}")
        print(f"         corners: {[[round(x, 2) for x in point] for point in quad.tolist()]}")
```

</hfoption>
</hfoptions>

### Batched inference

Pass a list of images and one target size per image. The reading order is decoded per image.

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers.image_utils import load_image


model_path = "PaddlePaddle/PP-DocLayoutV4_safetensors"
model = AutoModelForObjectDetection.from_pretrained(model_path, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = load_image("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg")
inputs = image_processor(images=[image, image], return_tensors="pt").to(model.device)

outputs = model(**inputs)
results = image_processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]] * 2)
for result in results:
    print("result:")
    for idx, (score, label_id, box) in enumerate(zip(result["scores"], result["labels"], result["boxes"])):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Order {idx + 1}: {model.config.id2label[label_id.item()]}, score: {score.item():.2f}, box: {box}")
```

## PPDocLayoutV4ForObjectDetection

[[autodoc]] PPDocLayoutV4ForObjectDetection
    - forward

## PPDocLayoutV4Model

[[autodoc]] PPDocLayoutV4Model

## PPDocLayoutV4Config

[[autodoc]] PPDocLayoutV4Config

## PPDocLayoutV4ImageProcessor

[[autodoc]] PPDocLayoutV4ImageProcessor
    - preprocess
    - post_process_object_detection
