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
*This model was released on 2022-01-10 and added to Hugging Face Transformers on 2022-02-07 and contributed by [nielsr](https://huggingface.co/nielsr).*

# ConvNeXT

[ConvNeXT](https://huggingface.co/papers/2201.03545) reexamines the design spaces of ConvNets and explores the potential of pure ConvNet architectures inspired by Vision Transformers. By modernizing a standard ResNet, the model identifies key components that enhance performance. ConvNeXT achieves competitive accuracy and scalability, reaching 87.8% ImageNet top-1 accuracy and outperforming Swin Transformers on COCO detection and ADE20K segmentation, while retaining the simplicity and efficiency of traditional ConvNets.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/convnext-tiny-224", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
model = AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## ConvNextConfig

[[autodoc]] ConvNextConfig

## ConvNextFeatureExtractor

[[autodoc]] ConvNextFeatureExtractor

## ConvNextImageProcessor

[[autodoc]] ConvNextImageProcessor
    - preprocess

## ConvNextImageProcessorFast

[[autodoc]] ConvNextImageProcessorFast
    - preprocess

## ConvNextModel

[[autodoc]] ConvNextModel
    - forward

## ConvNextForImageClassification

[[autodoc]] ConvNextForImageClassification
    - forward

