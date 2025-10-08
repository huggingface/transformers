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
*This model was released on 2023-01-02 and added to Hugging Face Transformers on 2023-03-14 and contributed by [adirik](https://huggingface.co/adirik).*

# ConvNeXt V2

[ConvNeXt V2](https://huggingface.co/papers/2301.00808) is a fully convolutional model inspired by Vision Transformers and built upon ConvNeXt. It integrates a novel Global Response Normalization (GRN) layer to enhance inter-channel feature competition and a fully convolutional masked autoencoder framework. This co-design improves performance on various recognition tasks, including ImageNet classification, COCO detection, and ADE20K segmentation. Pre-trained ConvNeXt V2 models range from an efficient 3.7M-parameter Atto model achieving 76.7% top-1 accuracy on ImageNet to a 650M Huge model with 88.9% accuracy using only public training data.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/convnextv2-tiny-1k-224", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-1k-224")
model = AutoModelForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## ConvNextV2Config

[[autodoc]] ConvNextV2Config

## ConvNextV2Model

[[autodoc]] ConvNextV2Model
    - forward

## ConvNextV2ForImageClassification

[[autodoc]] ConvNextV2ForImageClassification
    - forward

