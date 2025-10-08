<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-06-01 and added to Hugging Face Transformers on 2024-07-12 and contributed by [EduardoPacheco](https://huggingface.co/EduardoPacheco) and [namangarg110](https://huggingface.co/namangarg110).*

# Hiera

[Hiera](https://huggingface.co/papers/2306.00989) is a hierarchical Vision Transformer that simplifies the architecture by removing unnecessary vision-specific components, known as "bells-and-whistles," without sacrificing accuracy or efficiency. By pretraining with a strong visual pretext task (MAE), Hiera achieves superior performance and speed in both inference and training across various image and video recognition tasks. This model demonstrates that spatial biases can be effectively learned through proper pretraining, making additional architectural complexity redundant.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/hiera-base-224-hf", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("facebook/hiera-base-224-hf")
model = AutoModelForImageClassification.from_pretrained("facebook/hiera-base-224-hf", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## HieraConfig

[[autodoc]] HieraConfig

## HieraModel

[[autodoc]] HieraModel
    - forward

## HieraForPreTraining

[[autodoc]] HieraForPreTraining
    - forward
  
## HieraForImageClassification

[[autodoc]] HieraForImageClassification
    - forward

