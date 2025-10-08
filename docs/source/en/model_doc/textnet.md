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
*This model was released on 2021-11-03 and added to Hugging Face Transformers on 2025-01-08 and contributed by [Raghavan](https://huggingface.co/Raghavan), [jadechoghari](https://huggingface.co/jadechoghari), and [nielsr](https://huggingface.co/nielsr).*

# TextNet

[TextNet](https://huggingface.co/papers/2111.02394) introduces FAST, a scene text detection framework optimized for both speed and accuracy. Instead of relying on complex architectures and heavy post-processing, FAST uses a minimalist 1-channel kernel representation for arbitrary-shaped text and a GPU-parallel post-processing step that assembles text lines with minimal overhead. The network architecture is automatically searched and specialized for text detection, producing stronger features than classification-based backbones. FAST achieves state-of-the-art performance, reaching 81.6% F-measure at 152 FPS on Total-Text, and can be further accelerated to 600+ FPS with TensorRT, significantly outperforming prior fast detectors.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="czczup/textnet-base", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("czczup/textnet-base")
model = AutoModelForImageClassification.from_pretrained("czczup/textnet-base", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## TextNetConfig

[[autodoc]] TextNetConfig

## TextNetImageProcessor

[[autodoc]] TextNetImageProcessor
    - preprocess

## TextNetImageProcessorFast

[[autodoc]] TextNetImageProcessorFast
    - preprocess

## TextNetModel

[[autodoc]] TextNetModel
    - forward

## TextNetForImageClassification

[[autodoc]] TextNetForImageClassification
    - forward

