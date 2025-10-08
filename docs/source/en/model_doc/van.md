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
*This model was released on 2022-02-20 and added to Hugging Face Transformers on 2023-06-20 and contributed by [Francesco](https://huggingface.co/Francesco).*

> [!WARNING]
> This model is in maintenance mode only, we don’t accept any new PRs changing its code. If you run into any issues running this model, please reinstall the last version that supported this model: v4.30.0. You can do so by running the following command: pip install -U transformers==4.30.0.

# VAN

[Visual Attention Network](https://huggingface.co/papers/2202.09741) proposes a novel large kernel attention (LKA) module to address challenges in applying self-attention to images. This module enables self-adaptive and long-range correlations while avoiding issues related to 2D structure neglect, quadratic complexity, and lack of channel adaptability. The model, VAN, outperforms state-of-the-art vision transformers and convolutional neural networks across various tasks, including image classification, object detection, semantic segmentation, and instance segmentation.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="Visual-Attention-Network/van-base", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("Visual-Attention-Network/van-base")
model = AutoModelForImageClassification.from_pretrained("Visual-Attention-Network/van-base", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## VanConfig

[[autodoc]] VanConfig

## VanModel

[[autodoc]] VanModel
    - forward

## VanForImageClassification

[[autodoc]] VanForImageClassification
    - forward

