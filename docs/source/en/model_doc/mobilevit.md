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
*This model was released on 2021-10-05 and added to Hugging Face Transformers on 2022-06-29 and contributed by [Matthijs](https://huggingface.co/Matthijs).*

# MobileViT

[MobileViT](https://huggingface.co/papers/2110.02178) combines the strengths of convolutional neural networks and vision transformers to create a lightweight and low-latency model for mobile vision tasks. It introduces a novel layer that uses transformers for global processing instead of local convolutions. MobileViT achieves superior accuracy compared to CNNs and ViTs with a similar number of parameters, demonstrating its effectiveness on tasks like ImageNet-1k classification and MS-COCO object detection.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="apple/mobilevit-small", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
model = AutoModelForImageClassification.from_pretrained("apple/mobilevit-small", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## MobileViTConfig

[[autodoc]] MobileViTConfig

## MobileViTFeatureExtractor

[[autodoc]] MobileViTFeatureExtractor
    - __call__
    - post_process_semantic_segmentation

## MobileViTImageProcessor

[[autodoc]] MobileViTImageProcessor
    - preprocess
    - post_process_semantic_segmentation

## MobileViTImageProcessorFast

[[autodoc]] MobileViTImageProcessorFast
    - preprocess
    - post_process_semantic_segmentation

## MobileViTModel

[[autodoc]] MobileViTModel
    - forward

## MobileViTForImageClassification

[[autodoc]] MobileViTForImageClassification
    - forward

## MobileViTForSemanticSegmentation

[[autodoc]] MobileViTForSemanticSegmentation
    - forward

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="apple/mobilevit-small", dtype="auto")
pipeline("path/to/image.png")
```

