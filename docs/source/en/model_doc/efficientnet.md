<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2019-05-28 and added to Hugging Face Transformers on 2023-02-20 and contributed by [adirik](https://huggingface.co/adirik).*

# EfficientNet

[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://huggingface.co/papers/1905.11946) proposes a new scaling method for ConvNets by uniformly scaling network depth, width, and resolution using compound coefficients. This method is applied to design EfficientNets, a family of models that achieve superior accuracy and efficiency. EfficientNet-B7, for instance, attains 84.3% top-1 accuracy on ImageNet, being 8.4x smaller and 6.1x faster than the best existing ConvNets. The models also excel in transfer learning tasks, achieving state-of-the-art results with significantly fewer parameters.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="google/efficientnet-b7", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b7")
model = AutoModelForImageClassification.from_pretrained("google/efficientnet-b7", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## EfficientNetConfig

[[autodoc]] EfficientNetConfig

## EfficientNetImageProcessor

[[autodoc]] EfficientNetImageProcessor
    - preprocess

## EfficientNetImageProcessorFast

[[autodoc]] EfficientNetImageProcessorFast
    - preprocess

## EfficientNetModel

[[autodoc]] EfficientNetModel
    - forward

## EfficientNetForImageClassification

[[autodoc]] EfficientNetForImageClassification
    - forward

