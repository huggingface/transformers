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
*This model was released on 2021-03-29 and added to Hugging Face Transformers on 2022-05-18 and contributed by [anugunj](https://huggingface.co/anugunj).*

# Convolutional Vision Transformer (CvT)

[Convolutional vision Transformer (CvT)](https://huggingface.co/papers/2103.15808) enhances Vision Transformer (ViT) through the integration of convolutions, combining the strengths of both architectures. Key modifications include a hierarchical Transformer with a convolutional token embedding and a convolutional Transformer block with a convolutional projection. These enhancements introduce CNN properties like shift, scale, and distortion invariance while retaining Transformer benefits such as dynamic attention and global context. CvT achieves state-of-the-art performance on ImageNet-1k with fewer parameters and lower FLOPs, even when pretrained on larger datasets like ImageNet-22k. Notably, positional encoding can be omitted in CvT, simplifying the design for high-resolution vision tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="microsoft/cvt-13", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
model = AutoModelForImageClassification.from_pretrained("microsoft/cvt-13", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## CvtConfig

[[autodoc]] CvtConfig

## CvtModel

[[autodoc]] CvtModel
    - forward

## CvtForImageClassification

[[autodoc]] CvtForImageClassification
    - forward

