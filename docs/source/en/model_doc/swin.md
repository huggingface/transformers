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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
    </div>
</div>

# Swin Transformer

[Swin Transformer](https://huggingface.co/papers/2103.14030) is a hierarchical vision transformer. Images are processed in patches and windowed self-attention is used to capture local information. These windows are shifted across the image to allow for cross-window connections, capturing global information more efficiently. This hierarchical approach with shifted windows allows the Swin Transformer to process images effectively at different scales and achieve linear computational complexity relative to image size, making it a versatile backbone for various vision tasks like image classification and object detection.

You can find all official Swin Transformer checkpoints under the [Microsoft](https://huggingface.co/microsoft?search_models=swin) organization.

> [!TIP]
> Click on the Swin Transformer models in the right sidebar for more examples of how to apply Swin Transformer to different image tasks.

The example below demonstrates how to classify an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-classification",
    model="microsoft/swin-tiny-patch4-window7-224",
    dtype=torch.float16,
    device=0
)
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```
</hfoption>

<hfoption id="AutoModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224",
    use_fast=True,
)
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224",
    device_map="cuda"
)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(image, return_tensors="pt").to("cuda")

with torch.no_grad():
  logits = model(**inputs).logits
predicted_class_id = logits.argmax(dim=-1).item()

class_labels = model.config.id2label
predicted_class_label = class_labels[predicted_class_id]
print(f"The predicted class label is: {predicted_class_label}")
```
</hfoption>
</hfoptions>

## Notes

- Swin can pad the inputs for any input height and width divisible by `32`.
- Swin can be used as a [backbone](../backbones). When `output_hidden_states = True`, it outputs both `hidden_states` and `reshaped_hidden_states`. The `reshaped_hidden_states` have a shape of `(batch, num_channels, height, width)` rather than `(batch_size, sequence_length, num_channels)`.

## SwinConfig

[[autodoc]] SwinConfig

<frameworkcontent>
<pt>

## SwinModel

[[autodoc]] SwinModel
    - forward

## SwinForMaskedImageModeling

[[autodoc]] SwinForMaskedImageModeling
    - forward

## SwinForImageClassification

[[autodoc]] transformers.SwinForImageClassification
    - forward

</pt>
<tf>

## TFSwinModel

[[autodoc]] TFSwinModel
    - call

## TFSwinForMaskedImageModeling

[[autodoc]] TFSwinForMaskedImageModeling
    - call

## TFSwinForImageClassification

[[autodoc]] transformers.TFSwinForImageClassification
    - call

</tf>
</frameworkcontent>