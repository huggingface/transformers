<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# OmDet-Turbo

## Overview

The OmDet-Turbo model was proposed in [Real-time Transformer-based Open-Vocabulary Detection with Efficient Fusion Head](https://arxiv.org/abs/2403.06892) by Tiancheng Zhao, Peng Liu, Xuan He, Lu Zhang, Kyusong Lee. OmDet-Turbo incorporates components from RT-DETR and introduces a swift multimodal fusion module to achieve real-time open-vocabulary object detection capabilities while maintaining high accuracy. The base model achieves performance of up to 100.2 FPS and 53.4 AP on COCO zero-shot.

The abstract from the paper is the following:

*End-to-end transformer-based detectors (DETRs) have shown exceptional performance in both closed-set and open-vocabulary object detection (OVD) tasks through the integration of language modalities. However, their demanding computational requirements have hindered their practical application in real-time object detection (OD) scenarios. In this paper, we scrutinize the limitations of two leading models in the OVDEval benchmark, OmDet and Grounding-DINO, and introduce OmDet-Turbo. This novel transformer-based real-time OVD model features an innovative Efficient Fusion Head (EFH) module designed to alleviate the bottlenecks observed in OmDet and Grounding-DINO. Notably, OmDet-Turbo-Base achieves a 100.2 frames per second (FPS) with TensorRT and language cache techniques applied. Notably, in zero-shot scenarios on COCO and LVIS datasets, OmDet-Turbo achieves performance levels nearly on par with current state-of-the-art supervised models. Furthermore, it establishes new state-of-the-art benchmarks on ODinW and OVDEval, boasting an AP of 30.1 and an NMS-AP of 26.86, respectively. The practicality of OmDet-Turbo in industrial applications is underscored by its exceptional performance on benchmark datasets and superior inference speed, positioning it as a compelling choice for real-time object detection tasks.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/omdet_turbo_architecture.jpeg" alt="drawing" width="600"/>

<small> OmDet-Turbo architecture overview. Taken from the <a href="https://arxiv.org/abs/2403.06892">original paper</a>. </small>

This model was contributed by [yonigozlan](https://huggingface.co/yonigozlan).
The original code can be found [here](https://github.com/om-ai-lab/OmDet).

## Usage tips

One specificity of OmDet-Turbo compared to other zero-shot object detection models, such as [Grounding DINO](grounding-dino), is the decoupled classes and prompt embedding structure that allows caching of text embeddings. This means that the model needs both classes and task as inputs, where classes is a list of objects we want to detect and task is the grounded text used to guide open-vocabulary detection. This approach limits the scope of the open-vocabulary detection and makes the decoding process faster.

[`OmDetTurboProcessor`] is used to prepare the classes, task and image triplet. The task input is optional, and when not provided, it will default to `"Detect [class1], [class2], [class3], ..."` . To process the results from the model, one can use post_process_grounded_object_detection from [`OmDetTurboProcessor`]. Notably, this function takes in the input classes, as unlike other zero-shot object detection models, the decoupling of classes and task embeddings means that no decoding of the predicted class embeddings is needed in the post-processing step, and the predicted classes can be matched to the inputted ones directly.

## Usage example

### Single image inference

Here's how to load the model and prepare the inputs to perform zero-shot object detection on a single image:

```python
import requests
from PIL import Image

from transformers import AutoProcessor, OmDetTurboForObjectDetection

processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-tiny")
model = OmDetTurboForObjectDetection.from_pretrained("omlab/omdet-turbo-tiny")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
classes = ["cat", "remote"]
inputs = processor(image, text=classes, return_tensors="pt")

outputs = model(**inputs)

# convert outputs (bounding boxes and class logits)
results = processor.post_process_grounded_object_detection(
    outputs,
    classes=classes,
    target_sizes=[image.size[::-1]],
    score_threshold=0.3,
    nms_threshold=0.3,
)[0]
for score, class_name, box in zip(
    results["scores"], results["classes"], results["boxes"]
):
    box = [round(i, 1) for i in box.tolist()]
    print(
        f"Detected {class_name} with confidence "
        f"{round(score.item(), 2)} at location {box}"
    )
```

### Multi image inference

OmDet-Turbo can perform batched multi-image inference, with support for different text prompts and classes in the same batch:

```python
>>> import torch
>>> import requests
>>> from io import BytesIO
>>> from PIL import Image
>>> from transformers import AutoProcessor, OmDetTurboForObjectDetection
>>> processor = AutoProcessor.from_pretrained("yonigozlan/omdet-turbo-tiny")
>>> model = OmDetTurboForObjectDetection.from_pretrained("yonigozlan/omdet-turbo-tiny")
>>> url1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image1 = Image.open(BytesIO(requests.get(url1).content)).convert("RGB")
>>> classes1 = ["cat", "remote"]
>>> task1 = "Detect {}.".format(", ".join(classes1))
>>> url2 = "http://images.cocodataset.org/train2017/000000257813.jpg"
>>> image2 = Image.open(BytesIO(requests.get(url2).content)).convert("RGB")
>>> classes2 = ["boat"]
>>> task2 = "Detect all the boat in the image."
>>> url3 = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
>>> image3 = Image.open(BytesIO(requests.get(url3).content)).convert("RGB")
>>> classes3 = ["statue", "trees"]
>>> task3 = "Focus on the foreground, detect statue and trees."
>>> inputs = processor(
...     images=[image1, image2, image3],
...     text=[classes1, classes2, classes3],
...     task=[task1, task2, task3],
...     return_tensors="pt",
... )
>>> with torch.no_grad():
...     outputs = model(**inputs)
>>> # convert outputs (bounding boxes and class logits)
>>> results = processor.post_process_grounded_object_detection(
...     outputs,
...     classes=[classes1, classes2, classes3],
...     target_sizes=[image1.size[::-1], image2.size[::-1], image3.size[::-1]],
...     score_threshold=0.2,
...     nms_threshold=0.3,
... )
>>> for i, result in enumerate(results):
...     for score, class_name, box in zip(
...         result["scores"], result["classes"], result["boxes"]
...     ):
...         box = [round(i, 1) for i in box.tolist()]
...         print(
...             f"Detected {class_name} with confidence "
...             f"{round(score.item(), 2)} at location {box} in image {i}"
...         )
Detected remote with confidence 0.77 at location [39.9, 70.4, 176.7, 118.0] in image 0
Detected cat with confidence 0.72 at location [11.6, 54.2, 314.8, 474.0] in image 0
Detected remote with confidence 0.56 at location [333.4, 75.8, 370.7, 187.0] in image 0
Detected cat with confidence 0.55 at location [345.2, 24.0, 639.8, 371.7] in image 0
Detected boat with confidence 0.3 at location [146.5, 219.7, 209.7, 251.0] in image 1
Detected boat with confidence 0.27 at location [319.1, 222.1, 403.3, 239.2] in image 1
Detected boat with confidence 0.21 at location [37.7, 220.2, 84.0, 236.1] in image 1
Detected statue with confidence 0.73 at location [544.7, 210.2, 651.9, 502.8] in image 2
Detected trees with confidence 0.25 at location [3.9, 584.3, 391.4, 785.6] in image 2
Detected trees with confidence 0.25 at location [1.4, 621.2, 118.2, 787.8] in image 2
Detected statue with confidence 0.2 at location [428.1, 205.5, 767.3, 759.5] in image 2

```

## OmDetTurboConfig

[[autodoc]] OmDetTurboConfig

## OmDetTurboProcessor

[[autodoc]] OmDetTurboProcessor
    - post_process_grounded_object_detection

## OmDetTurboForObjectDetection

[[autodoc]] OmDetTurboForObjectDetection
    - forward
