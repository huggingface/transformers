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
*This model was released on 2022-06-02 and added to Hugging Face Transformers on 2023-06-20 and contributed by [novice03](https://huggingface.co/novice03) and [Bearnardd](https://huggingface.co/Bearnardd).*

# EfficientFormer

[EfficientFormer: Vision Transformers at MobileNet Speed](https://huggingface.co/papers/2206.01191) proposes a dimension-consistent pure transformer designed for mobile devices, addressing the speed limitations of traditional Vision Transformers (ViT). By revisiting and optimizing the architecture and operators used in ViT-based models, EfficientFormer achieves high performance with low latency. Experiments demonstrate that EfficientFormer-L1 matches MobileNetV2×1.4 in speed (1.6 ms inference latency on iPhone 12) while achieving 79.2% top-1 accuracy on ImageNet-1K. The largest model, EfficientFormer-L7, reaches 83.3% accuracy with 7.0 ms latency, proving that transformers can be effectively deployed on mobile devices for real-time applications.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="snap-research/efficientformer-l1-300", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("snap-research/efficientformer-l1-300")
model = AutoModelForImageClassification.from_pretrained("snap-research/efficientformer-l1-300", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## EfficientFormerConfig

[[autodoc]] EfficientFormerConfig

## EfficientFormerImageProcessor

[[autodoc]] EfficientFormerImageProcessor
    - preprocess

## EfficientFormerModel

[[autodoc]] EfficientFormerModel
    - forward

## EfficientFormerForImageClassification

[[autodoc]] EfficientFormerForImageClassification
    - forward

## EfficientFormerForImageClassificationWithTeacher

[[autodoc]] EfficientFormerForImageClassificationWithTeacher
    - forward

