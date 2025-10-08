<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2021-06-25 and added to Hugging Face Transformers on 2024-03-13 and contributed by [FoamoftheSea](https://huggingface.co/FoamoftheSea).*

# Pyramid Vision Transformer V2 (PVTv2)

Pyramid Vision Transformer v2](https://huggingface.co/papers/2102.12122) enhances the original PVT by introducing a linear complexity attention layer, overlapping patch embedding, and convolutional feed-forward network. These improvements reduce computational complexity to linear levels and boost performance on classification, detection, and segmentation tasks, matching or surpassing recent models like Swin Transformer.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="OpenGVLab/pvt_v2_b0", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("OpenGVLab/pvt_v2_b0")
model = AutoModelForImageClassification.from_pretrained("OpenGVLab/pvt_v2_b0", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## PvtV2Config

[[autodoc]] PvtV2Config

## PvtForImageClassification

[[autodoc]] PvtV2ForImageClassification
    - forward

## PvtModel

[[autodoc]] PvtV2Model
    - forward
