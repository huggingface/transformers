<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2021-02-24 and added to Hugging Face Transformers on 2023-07-24 and contributed by [Xrenya](https://huggingface.co/Xrenya).*

# Pyramid Vision Transformer (PVT)

[Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://huggingface.co/papers/2102.12122) proposes PVT, a vision transformer with a pyramid structure for dense prediction tasks. PVT uses fine-grained inputs and progressively shrinks feature maps to reduce computational costs. It incorporates a spatial-reduction attention layer to further decrease resource consumption for high-resolution features. PVT combines the strengths of CNNs and Transformers, serving as a versatile backbone for various vision tasks without convolutions. Experiments demonstrate that PVT enhances performance in object detection, instance, and semantic segmentation, outperforming ResNet50+RetinNet on the COCO dataset.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="Xrenya/pvt-tiny-224", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("Xrenya/pvt-tiny-224")
model = AutoModelForImageClassification.from_pretrained("Xrenya/pvt-tiny-224", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## PvtConfig

[[autodoc]] PvtConfig

## PvtImageProcessor

[[autodoc]] PvtImageProcessor
    - preprocess

## PvtImageProcessorFast

[[autodoc]] PvtImageProcessorFast
    - preprocess

## PvtForImageClassification

[[autodoc]] PvtForImageClassification
    - forward

## PvtModel

[[autodoc]] PvtModel
    - forward

