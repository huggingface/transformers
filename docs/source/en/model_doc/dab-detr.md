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

# DAB-DETR

## Overview

The DAB-DETR model was proposed in [DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR](https://arxiv.org/abs/2201.12329) by Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, Lei Zhang.
DAB-DETR is an enhanced variant of Conditional DETR. It utilizes dynamically updated anchor boxes to provide both a reference query point (x, y) and a reference anchor size (w, h), improving cross-attention computation. This new approach achieves 45.7% AP when trained for 50 epochs with a single ResNet-50 model as the backbone.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dab_detr_convergence_plot.png"
alt="drawing" width="600"/>

The abstract from the paper is the following:

*We present in this paper a novel query formulation using dynamic anchor boxes
for DETR (DEtection TRansformer) and offer a deeper understanding of the role
of queries in DETR. This new formulation directly uses box coordinates as queries
in Transformer decoders and dynamically updates them layer-by-layer. Using box
coordinates not only helps using explicit positional priors to improve the query-to-feature similarity and eliminate the slow training convergence issue in DETR,
but also allows us to modulate the positional attention map using the box width
and height information. Such a design makes it clear that queries in DETR can be
implemented as performing soft ROI pooling layer-by-layer in a cascade manner.
As a result, it leads to the best performance on MS-COCO benchmark among
the DETR-like detection models under the same setting, e.g., AP 45.7% using
ResNet50-DC5 as backbone trained in 50 epochs. We also conducted extensive
experiments to confirm our analysis and verify the effectiveness of our methods.*

This model was contributed by [davidhajdu](https://huggingface.co/davidhajdu).
The original code can be found [here](https://github.com/IDEA-Research/DAB-DETR).

## How to Get Started with the Model

Use the code below to get started with the model.

```python
import torch
import requests

from PIL import Image
from transformers import AutoModelForObjectDetection, AutoImageProcessor

url = 'http://images.cocodataset.org/val2017/000000039769.jpg' 
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("IDEA-Research/dab-detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("IDEA-Research/dab-detr-resnet-50")

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)

for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{model.config.id2label[label]}: {score:.2f} {box}")
```
This should output
```
cat: 0.87 [14.7, 49.39, 320.52, 469.28]
remote: 0.86 [41.08, 72.37, 173.39, 117.2]
cat: 0.86 [344.45, 19.43, 639.85, 367.86]
remote: 0.61 [334.27, 75.93, 367.92, 188.81]
couch: 0.59 [-0.04, 1.34, 639.9, 477.09]
```

There are three other ways to instantiate a DAB-DETR model (depending on what you prefer):

Option 1: Instantiate DAB-DETR with pre-trained weights for entire model
```py
>>> from transformers import DabDetrForObjectDetection

>>> model = DabDetrForObjectDetection.from_pretrained("IDEA-Research/dab-detr-resnet-50")
```

Option 2: Instantiate DAB-DETR with randomly initialized weights for Transformer, but pre-trained weights for backbone
```py
>>> from transformers import DabDetrConfig, DabDetrForObjectDetection

>>> config = DabDetrConfig()
>>> model = DabDetrForObjectDetection(config)
```
Option 3: Instantiate DAB-DETR with randomly initialized weights for backbone + Transformer
```py
>>> config = DabDetrConfig(use_pretrained_backbone=False)
>>> model = DabDetrForObjectDetection(config)
```


## DabDetrConfig

[[autodoc]] DabDetrConfig

## DabDetrModel

[[autodoc]] DabDetrModel
    - forward

## DabDetrForObjectDetection

[[autodoc]] DabDetrForObjectDetection
    - forward
