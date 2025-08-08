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
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# MobileNet V2

[MobileNet V2](https://huggingface.co/papers/1801.04381) improves performance on mobile devices with a more efficient architecture. It uses inverted residual blocks and linear bottlenecks to start with a smaller representation of the data, expands it for processing, and shrinks it again to reduce the number of computations. The model also removes non-linearities to maintain accuracy despite its simplified design. Like [MobileNet V1](./mobilenet_v1), it uses depthwise separable convolutions for efficiency.

You can all the original MobileNet checkpoints under the [Google](https://huggingface.co/google?search_models=mobilenet) organization.

> [!TIP]
> Click on the MobileNet V2 models in the right sidebar for more examples of how to apply MobileNet to different vision tasks.


The examples below demonstrate how to classify an image with [`Pipeline`] or the [`AutoModel`] class.


<hfoptions id="usage-img-class">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-classification",
    model="google/mobilenet_v2_1.4_224",
    dtype=torch.float16,
    device=0
)
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
import requests
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained(
    "google/mobilenet_v2_1.4_224",
)
model = AutoModelForImageClassification.from_pretrained(
    "google/mobilenet_v2_1.4_224",
)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(image, return_tensors="pt")

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

-   Classification checkpoint names follow the pattern `mobilenet_v2_{depth_multiplier}_{resolution}`, like `mobilenet_v2_1.4_224`. `1.4` is the depth multiplier and `224` is the image resolution. Segmentation checkpoint names follow the pattern `deeplabv3_mobilenet_v2_{depth_multiplier}_{resolution}`.
-   While trained on images of a specific sizes, the model architecture works with images of different sizes (minimum 32x32). The [`MobileNetV2ImageProcessor`] handles the necessary preprocessing.
-   MobileNet is pretrained on [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k), a dataset with 1000 classes. However, the model actually predicts 1001 classes. The additional class is an extra "background" class (index 0).
-   The segmentation models use a [DeepLabV3+](https://huggingface.co/papers/1802.02611) head which is often pretrained on datasets like [PASCAL VOC](https://huggingface.co/datasets/merve/pascal-voc).
-   The original TensorFlow checkpoints determines the padding amount at inference because it depends on the input image size. To use the native PyTorch padding behavior, set `tf_padding=False` in [`MobileNetV2Config`].
    ```python
    from transformers import MobileNetV2Config

    config = MobileNetV2Config.from_pretrained("google/mobilenet_v2_1.4_224", tf_padding=True)
    ```
-   The Transformers implementation does not support the following features.
    -   Uses global average pooling instead of the optional 7x7 average pooling with stride 2. For larger inputs, this gives a pooled output that is larger than a 1x1 pixel.
    -   `output_hidden_states=True` returns *all* intermediate hidden states. It is not possible to extract the output from specific layers for other downstream purposes.
    - Does not include the quantized models from the original checkpoints because they include "FakeQuantization" operations to unquantize the weights.
    -   For segmentation models, the final convolution layer of the backbone is computed even though the DeepLabV3+ head doesn't use it.

## MobileNetV2Config

[[autodoc]] MobileNetV2Config

## MobileNetV2FeatureExtractor

[[autodoc]] MobileNetV2FeatureExtractor
    - preprocess
    - post_process_semantic_segmentation

## MobileNetV2ImageProcessor

[[autodoc]] MobileNetV2ImageProcessor
    - preprocess
    - post_process_semantic_segmentation

## MobileNetV2ImageProcessorFast

[[autodoc]] MobileNetV2ImageProcessorFast
    - preprocess
    - post_process_semantic_segmentation

## MobileNetV2Model

[[autodoc]] MobileNetV2Model
    - forward

## MobileNetV2ForImageClassification

[[autodoc]] MobileNetV2ForImageClassification
    - forward

## MobileNetV2ForSemanticSegmentation

[[autodoc]] MobileNetV2ForSemanticSegmentation
    - forward
