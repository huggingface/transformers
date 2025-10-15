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
*This model was released on 2019-12-24 and added to Hugging Face Transformers on 2022-12-07 and contributed by [nielsr](https://huggingface.co/nielsr).*

# Big Transfer (BiT)

[Big Transfer (BiT): General Visual Representation Learning](https://huggingface.co/papers/1912.11370) proposes a method for scaling up pre-training of ResNetv2 architectures. This approach, called Big Transfer (BiT), combines specific components and uses a simple heuristic for transfer learning, achieving strong performance across over 20 datasets. BiT demonstrates robustness across various data regimes, from 1 example per class to 1M total examples. It achieves 87.5% top-1 accuracy on ILSVRC-2012, 99.4% on CIFAR-10, and 76.3% on the 19-task Visual Task Adaptation Benchmark (VTAB). On small datasets, BiT reaches 76.8% on ILSVRC-2012 with 10 examples per class and 97.0% on CIFAR-10 with 10 examples per class. The paper includes a detailed analysis of the key components contributing to high transfer performance.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="google/bit-50", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("google/bit-50")
model = AutoModelForImageClassification.from_pretrained("google/bit-50", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## Usage tips

- BiT models are equivalent to ResNetv2 in architecture with two key differences: all batch normalization layers are replaced by group normalization, and weight standardization is used for convolutional layers.
- The combination of group normalization and weight standardization is useful for training with large batch sizes and has a significant impact on transfer learning.

## BitConfig

[[autodoc]] BitConfig

## BitImageProcessor

[[autodoc]] BitImageProcessor
    - preprocess

## BitImageProcessorFast

[[autodoc]] BitImageProcessorFast
    - preprocess

## BitModel

[[autodoc]] BitModel
    - forward

## BitForImageClassification

[[autodoc]] BitForImageClassification
    - forward

