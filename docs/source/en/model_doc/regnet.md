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
*This model was released on 2020-03-30 and added to Hugging Face Transformers on 2022-04-07 and contributed by [Francesco](https://huggingface.co/Francesco).*

# RegNet

[RegNet](https://huggingface.co/papers/2003.13678) presents a novel network design paradigm by focusing on network design spaces rather than individual network instances. Through iterative reduction of a high-dimensional search space, the authors identify a low-dimensional design space of simple, regular networks called RegNet. The key insight is that the widths and depths of effective networks can be described by a quantized linear function. This design space leads to networks that are fast and perform well across various computational budgets, outperforming EfficientNet models while being up to 5x faster on GPUs.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/regnet-y-040", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("facebook/regnet-y-040")
model = AutoModelForImageClassification.from_pretrained("facebook/regnet-y-040", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>
## RegNetConfig

[[autodoc]] RegNetConfig

## RegNetModel

[[autodoc]] RegNetModel
    - forward

## RegNetForImageClassification

[[autodoc]] RegNetForImageClassification
    - forward

