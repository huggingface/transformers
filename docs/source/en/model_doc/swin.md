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
*This model was released on 2021-03-25 and added to Hugging Face Transformers on 2022-01-21 and contributed by [novice03](https://huggingface.co/novice03).*

# Swin Transformer

[Swin Transformer](https://huggingface.co/papers/2103.14030) presents a hierarchical vision Transformer using shifted windows to address challenges in adapting Transformer models from language to vision. The shifted windowing scheme enhances efficiency by focusing self-attention on non-overlapping local windows while enabling cross-window connections. This design supports multi-scale modeling with linear computational complexity relative to image size. Swin Transformer excels in various vision tasks, achieving top-1 accuracy of 87.3% on ImageNet-1K, 58.7 box AP and 51.1 mask AP on COCO, and 53.5 mIoU on ADE20K. Its performance significantly outperforms previous state-of-the-art models, highlighting the potential of Transformer-based architectures in vision.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="microsoft/swin-tiny-patch4-window7-224", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = AutoModelForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## SwinConfig

[[autodoc]] SwinConfig

## SwinModel

[[autodoc]] SwinModel
    - forward

## SwinForMaskedImageModeling

[[autodoc]] SwinForMaskedImageModeling
    - forward

## SwinForImageClassification

[[autodoc]] transformers.SwinForImageClassification
    - forward

