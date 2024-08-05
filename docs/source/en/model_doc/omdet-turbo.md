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

One specificity of OmDet-Turbo compared to other zero-shot object detection models, such as [Grounding DINO](grounding-dino), is the decoupled classes and prompt embedding structure that allows caching of text embeddings. This means that the user needs to provide both text and classes as inputs, where text is the grounded text used to guide open-vocabulary detection, and classes is a list of the objects we want to detect. This approach limits the scope of the open-vocabulary detection and makes the decoding process faster.

[`OmDetTurboProcessor`] is used to prepare the text, classes, and image triplet. To process the results from the model, one can use post_process_grounded_object_detection from [`OmDetTurboProcessor`]. Notably, this function takes in the input classes, as unlike other zero-shot object detection models, the decoupling of classes and prompt embeddings means that no decoding of the predicted class embeddings is needed in the post-processing step, and the predicted classes can be matched to the inputted ones directly.

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
task = "Detect {}.".format(",".join(classes))
inputs = processor(image, text=task, classes=classes, return_tensors="pt")

outputs = model(**inputs)

# convert outputs (bounding boxes and class logits)
results = processor.post_process_grounded_object_detection(
    outputs,
    classes=[classes],
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
from io import BytesIO

import requests
from PIL import Image

from transformers import AutoProcessor, OmDetTurboForObjectDetection

processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-tiny")
model = OmDetTurboForObjectDetection.from_pretrained("omlab/omdet-turbo-tiny")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
url2 = "http://images.cocodataset.org/train2017/000000257813.jpg"
url3 = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
image1 = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
image2 = Image.open(BytesIO(requests.get(url2).content)).convert("RGB")
image3 = Image.open(BytesIO(requests.get(url3).content)).convert("RGB")
classes1 = ["cat", "remote"]
prompt1 = "Detect {}.".format(",".join(classes1))
classes2 = ["boat"]
prompt2 = "Detect all the boat in the image."
classes3 = ["statue", "trees"]
prompt3 = "Focus on the foreground, detect statue and trees."
inputs = processor(
    images=[image1, image2, image3],
    text=[prompt1, prompt2, prompt3],
    classes=[classes1, classes2, classes3],
    return_tensors="pt",
)

outputs = model(**inputs)

# convert outputs (bounding boxes and class logits)
results = processor.post_process_grounded_object_detection(
    outputs,
    classes=[classes1, classes2, classes3],
    target_sizes=[image1.size[::-1], image2.size[::-1], image3.size[::-1]],
    score_threshold=0.2,
    nms_threshold=0.3,
)
for i, result in enumerate(results):
    for score, class_name, box in zip(
        result["scores"], result["classes"], result["boxes"]
    ):
        box = [round(i, 1) for i in box.tolist()]
        print(
            f"Detected {class_name} with confidence "
            f"{round(score.item(), 2)} at location {box} in image {i}"
        )

```

## OmDetTurboConfig

[[autodoc]] OmDetTurboConfig

## OmDetTurboProcessor

[[autodoc]] OmDetTurboProcessor
    - post_process_grounded_object_detection

## OmDetTurboForObjectDetection

[[autodoc]] OmDetTurboForObjectDetection
    - forward
