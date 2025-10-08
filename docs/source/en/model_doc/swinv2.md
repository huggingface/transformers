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
*This model was released on 2021-11-18 and added to Hugging Face Transformers on 2022-07-27 and contributed by [nandwalritik](https://huggingface.co/nandwalritik).*

# Swin Transformer V2

[Swin Transformer V2](https://huggingface.co/papers/2111.09883) addresses training instability, resolution gaps, and data hunger in large vision models. It introduces a residual-post-norm method with cosine attention, a log-spaced continuous position bias, and a self-supervised pre-training method called SimMIM. These techniques enabled the training of a 3 billion-parameter model capable of handling 1,536×1,536 resolution images, setting new benchmarks in ImageNet-V2 classification, COCO detection, ADE20K segmentation, and Kinetics-400 action classification. The training process is more efficient, using 40 times less labeled data and time compared to similar models.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="microsoft/swinv2-tiny-patch4-window8-256", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## Swinv2Config

[[autodoc]] Swinv2Config

## Swinv2Model

[[autodoc]] Swinv2Model
    - forward

## Swinv2ForMaskedImageModeling

[[autodoc]] Swinv2ForMaskedImageModeling
    - forward

## Swinv2ForImageClassification

[[autodoc]] transformers.Swinv2ForImageClassification
    - forward

