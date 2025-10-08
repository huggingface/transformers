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
*This model was released on 2017-04-17 and added to Hugging Face Transformers on 2022-11-21 and contributed by [Matthijs](https://huggingface.co/Matthijs).*

# MobileNet V1

[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://huggingface.co/papers/1704.04861) presents MobileNets, a class of efficient models for mobile and embedded vision applications. These models use depth-wise separable convolutions to create lightweight deep neural networks. Two global hyper-parameters are introduced to balance latency and accuracy, enabling model builders to select the appropriate model size for specific applications. Experiments demonstrate strong performance on ImageNet classification and across various applications such as object detection, fine-grain classification, face attributes, and large-scale geo-localization.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="google/mobilenet_v1_1.0_224", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("google/mobilenet_v1_1.0_224")
model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v1_1.0_224", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## MobileNetV1Config

[[autodoc]] MobileNetV1Config

## MobileNetV1FeatureExtractor

[[autodoc]] MobileNetV1FeatureExtractor
    - preprocess

## MobileNetV1ImageProcessor

[[autodoc]] MobileNetV1ImageProcessor
    - preprocess

## MobileNetV1ImageProcessorFast

[[autodoc]] MobileNetV1ImageProcessorFast
    - preprocess

## MobileNetV1Model

[[autodoc]] MobileNetV1Model
    - forward

## MobileNetV1ForImageClassification

[[autodoc]] MobileNetV1ForImageClassification
    - forward
