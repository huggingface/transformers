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
*This model was released on 2018-01-13 and added to Hugging Face Transformers on 2022-11-14 and contributed by [Matthijs](https://huggingface.co/Matthijs).*

# MobileNet V2

[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://huggingface.co/papers/1801.04381) describes a new mobile architecture that enhances the performance of mobile models across various tasks and sizes. It introduces an inverted residual structure with thin bottleneck layers and lightweight depthwise convolutions. The model emphasizes the importance of removing non-linearities in narrow layers to preserve representational power. This design allows for decoupling the input/output domains from the transformation's expressiveness. Performance is evaluated on Imagenet classification, COCO object detection, and VOC image segmentation, with a focus on balancing accuracy, multiply-adds, and parameters.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="google/mobilenet_v2_1.4_224", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.4_224")
model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.4_224", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## MobileNetV2Config

[[autodoc]] MobileNetV2Config

## MobileNetV2ImageProcessor

[[autodoc]] MobileNetV2ImageProcessor
    - preprocess
    - post_process_semantic_segmentation

## MobileNetV2ImageProcessorFast

[[autodoc]] MobileNetV2ImageProcessorFast
    - preprocess
    - post_process_semantic_segmentation

## MobileNetV2Model

[[autodoc]] MobileNetV2Model
    - forward

## MobileNetV2ForImageClassification

[[autodoc]] MobileNetV2ForImageClassification
    - forward

## MobileNetV2ForSemanticSegmentation

[[autodoc]] MobileNetV2ForSemanticSegmentation
    - forward