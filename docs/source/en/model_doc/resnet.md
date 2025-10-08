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
*This model was released on 2015-12-10 and added to Hugging Face Transformers on 2022-03-14 and contributed by [Francesco](https://huggingface.co/Francesco).*

# ResNet

[ResNet](https://huggingface.co/papers/1512.03385) introduced residual connections to facilitate the training of very deep neural networks. By learning residual functions relative to the layer inputs, ResNet enables the training of networks with up to 152 layers, significantly deeper than previous models. This approach improves optimization and accuracy. On the ImageNet dataset, ResNet achieved a 3.57% error rate with an ensemble of deep residual networks, winning first place in the ILSVRC 2015 classification task. Additionally, ResNet demonstrated a 28% relative improvement on the COCO object detection dataset, securing top positions in multiple ILSVRC & COCO 2015 competitions.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="microsoft/resnet-50", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## ResNetConfig

[[autodoc]] ResNetConfig

## ResNetModel

[[autodoc]] ResNetModel
    - forward

## ResNetForImageClassification

[[autodoc]] ResNetForImageClassification
    - forward

