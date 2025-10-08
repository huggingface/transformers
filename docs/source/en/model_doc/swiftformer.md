<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-03-27 and added to Hugging Face Transformers on 2023-05-12 and contributed by [shehan97](https://huggingface.co/shehan97).*

# SwiftFormer

[SwiftFormer](https://huggingface.co/papers/2303.15446) introduces an efficient additive attention mechanism that replaces quadratic matrix multiplications in self-attention with linear element-wise multiplications, enabling its use throughout the network without accuracy loss. This results in a series of models achieving top performance in accuracy and mobile inference speed. The small variant of SwiftFormer achieves 78.5% top-1 ImageNet-1K accuracy with 0.8 ms latency on iPhone 14, surpassing MobileViT-v2 in both accuracy and speed.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="MBZUAI/swiftformer-xs", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("MBZUAI/swiftformer-xs")
model = AutoModelForImageClassification.from_pretrained("MBZUAI/swiftformer-xs", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## SwiftFormerConfig

[[autodoc]] SwiftFormerConfig

## SwiftFormerModel

[[autodoc]] SwiftFormerModel
    - forward

## SwiftFormerForImageClassification

[[autodoc]] SwiftFormerForImageClassification
    - forward

