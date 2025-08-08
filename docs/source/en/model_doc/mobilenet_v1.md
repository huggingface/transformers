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

# MobileNet V1

[MobileNet V1](https://huggingface.co/papers/1704.04861) is a family of efficient convolutional neural networks optimized for on-device or embedded vision tasks. It achieves this efficiency by using depth-wise separable convolutions instead of standard convolutions. The architecture allows for easy trade-offs between latency and accuracy using two main hyperparameters, a width multiplier (alpha) and an image resolution multiplier.

You can all the original MobileNet checkpoints under the [Google](https://huggingface.co/google?search_models=mobilenet) organization.

> [!TIP]
> Click on the MobileNet V1 models in the right sidebar for more examples of how to apply MobileNet to different vision tasks.

The example below demonstrates how to classify an image with [`Pipeline`] or the [`AutoModel`] class.


<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-classification",
    model="google/mobilenet_v1_1.0_224",
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
    "google/mobilenet_v1_1.0_224",
)
model = AutoModelForImageClassification.from_pretrained(
    "google/mobilenet_v1_1.0_224",
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

<!-- Quantization - Not applicable -->
<!-- Attention Visualization - Not applicable for this model type -->


## Notes

-   Checkpoint names follow the pattern `mobilenet_v1_{depth_multiplier}_{resolution}`, like `mobilenet_v1_1.0_224`. `1.0` is the depth multiplier and `224` is the image resolution.
-   While trained on images of a specific sizes, the model architecture works with images of different sizes (minimum 32x32). The [`MobileNetV1ImageProcessor`] handles the necessary preprocessing.
-   MobileNet is pretrained on [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k), a dataset with 1000 classes. However, the model actually predicts 1001 classes. The additional class is an extra "background" class (index 0).
-   The original TensorFlow checkpoints determines the padding amount at inference because it depends on the input image size. To use the native PyTorch padding behavior, set `tf_padding=False` in [`MobileNetV1Config`].
    ```python
    from transformers import MobileNetV1Config

    config = MobileNetV1Config.from_pretrained("google/mobilenet_v1_1.0_224", tf_padding=True)
    ```
-   The Transformers implementation does not support the following features.
    -   Uses global average pooling instead of the optional 7x7 average pooling with stride 2. For larger inputs, this gives a pooled output that is larger than a 1x1 pixel.
    -   Does not support other `output_stride` values (fixed at 32). For smaller `output_strides`, the original implementation uses dilated convolution to prevent spatial resolution from being reduced further. (which would require dilated convolutions).
    -   `output_hidden_states=True` returns *all* intermediate hidden states. It is not possible to extract the output from specific layers for other downstream purposes.
    - Does not include the quantized models from the original checkpoints because they include "FakeQuantization" operations to unquantize the weights.

## MobileNetV1Config

[[autodoc]] MobileNetV1Config

## MobileNetV1FeatureExtractor

[[autodoc]] MobileNetV1FeatureExtractor
    - preprocess

## MobileNetV1ImageProcessor

[[autodoc]] MobileNetV1ImageProcessor
    - preprocess

## MobileNetV1ImageProcessorFast

[[autodoc]] MobileNetV1ImageProcessorFast
    - preprocess

## MobileNetV1Model

[[autodoc]] MobileNetV1Model
    - forward

## MobileNetV1ForImageClassification

[[autodoc]] MobileNetV1ForImageClassification
    - forward
