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
*This model was released on 2021-11-22 and added to Hugging Face Transformers on 2022-02-17 and contributed by [heytanay](https://huggingface.co/heytanay).*

# PoolFormer

[PoolFormer](https://huggingface.co/papers/2111.11418) demonstrates that the general architecture of transformers, rather than the specific token mixer module, is crucial for performance in computer vision tasks. By replacing the attention module with a simple spatial pooling operator, PoolFormer achieves competitive accuracy on ImageNet-1K with significantly fewer parameters and computations compared to well-tuned vision transformer and MLP-like baselines. This work introduces the concept of MetaFormer, a general architecture abstracted from transformers, and suggests that future research should focus on improving MetaFormer rather than refining token mixer modules.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="sail/poolformer_s12", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("sail/poolformer_s12")
model = AutoModelForImageClassification.from_pretrained("sail/poolformer_s12", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## PoolFormerConfig

[[autodoc]] PoolFormerConfig

## PoolFormerImageProcessor

[[autodoc]] PoolFormerImageProcessor
    - preprocess

## PoolFormerImageProcessorFast

[[autodoc]] PoolFormerImageProcessorFast
    - preprocess

## PoolFormerModel

[[autodoc]] PoolFormerModel
    - forward

## PoolFormerForImageClassification

[[autodoc]] PoolFormerForImageClassification
    - forward

