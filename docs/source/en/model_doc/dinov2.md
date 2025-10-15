<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2023-04-14 and added to Hugging Face Transformers on 2023-07-18 and contributed by [nielsr](https://huggingface.co/nielsr).*

# DINOv2

[DINOv2: Learning Robust Visual Features without Supervision](https://huggingface.co/papers/2304.07193) revisits and combines existing self-supervised techniques to produce all-purpose visual features through large-scale pretraining. It introduces an automatic pipeline for building a diverse and curated image dataset, and trains a Vision Transformer with 1B parameters, which is then distilled into smaller models. These models outperform OpenCLIP on most benchmarks at both image and pixel levels.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/dinov2-small-imagenet1k-1-layer", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small-imagenet1k-1-layer")
model = AutoModelForImageClassification.from_pretrained("facebook/dinov2-small-imagenet1k-1-layer", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## Usage tips

- Use [torch.jit.trace](https://docs.pytorch.org/docs/stable/generated/torch.jit.trace.html) to speed up inference. However, it produces some mismatched elements. The difference between the original and traced model is 1e-4.

## Dinov2Config

[[autodoc]] Dinov2Config

## Dinov2Model

[[autodoc]] Dinov2Model
    - forward

## Dinov2ForImageClassification

[[autodoc]] Dinov2ForImageClassification
    - forward

