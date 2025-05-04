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

<!-- Floating div for badges, image classification task -->
<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white">
        <a href="https://huggingface.co/docs/transformers/tasks/image_classification">
            <img alt="Image Classification" src="https://img.shields.io/badge/Task-Image%20Classification-yellow">
        </a>
        <!-- Add TF/Flax badges if supported -->
    </div>
</div>

# MobileNet V1

MobileNet V1 is a family of efficient convolutional neural networks optimized for on-device or embedded vision tasks. It achieves this efficiency by using depth-wise separable convolutions instead of standard convolutions. The architecture allows for easy trade-offs between latency and accuracy using two main hyperparameters: a width multiplier (alpha) and an image resolution multiplier.

You can find checkpoints like [`google/mobilenet_v1_1.0_224`](https://huggingface.co/google/mobilenet_v1_1.0_224), [`google/mobilenet_v1_0.75_192`](https://huggingface.co/google/mobilenet_v1_0.75_192) on the Hub.

> [!TIP]
> Click on the MobileNet V1 models in the right sidebar for more examples of how to apply MobileNet to different vision tasks.



The example below demonstrates how to perform image classification using [`pipeline`] or [`AutoModelForImageClassification`].


<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import requests
from PIL import Image
from transformers import pipeline

# Use a pipeline as a high-level helper
pipe = pipeline("image-classification", model="google/mobilenet_v1_1.0_224")

# Load image from the web
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Pass image to the pipeline
results = pipe(image)
print(results)
# Example output: [{'score': 0.963..., 'label': 'tabby, tabby cat'}, ...]
```

</hfoption>
<hfoption id="AutoModel">

```python
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load processor and model
processor = AutoImageProcessor.from_pretrained("google/mobilenet_v1_1.0_224")
model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v1_1.0_224")

# Load image from the web
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Preprocess image
inputs = processor(images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Post-process to get labels
predicted_label_id = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_label_id]
print(predicted_label)
# Example output: tabby, tabby cat
```

</hfoption>
</hfoptions>

<!-- Quantization section omitted - HF checkpoints not quantized per original docs -->
<!-- Attention Visualization section omitted - Not applicable for this model type -->


## Notes

-   **Checkpoint Naming:** Checkpoints often follow the pattern `mobilenet_v1_{depth_multiplier}_{resolution}`, like `mobilenet_v1_1.0_224`.
-   **Variable Input Size:** While trained on fixed sizes (e.g., 224x224), the model architecture works with images of different sizes (minimum 32x32). The [`MobileNetV1ImageProcessor`] handles the necessary preprocessing.
-   **1001 Classes:** Models pretrained on ImageNet-1k output 1001 classes. Index 0 is a "background" class, and indices 1-1000 correspond to the standard ImageNet classes.
-   **Padding Differences:** Original TensorFlow checkpoints had dynamic padding behavior based on input size. The Hugging Face PyTorch implementation uses standard, static padding by default. To enable dynamic padding (matching TensorFlow behavior), instantiate the configuration with `tf_padding=True`.
    ```python
    from transformers import MobileNetV1Config

    # Example: Load config with dynamic padding enabled
    config = MobileNetV1Config.from_pretrained("google/mobilenet_v1_1.0_224", tf_padding=True)
    ```
-   **Unsupported Features:** The Hugging Face implementation has some differences from the original paper/TF implementation:
    -   Uses global average pooling instead of the optional 7x7 average pooling with stride 2.
    -   Does not support specifying a custom `output_stride` (which would require dilated convolutions). The output stride is fixed at 32.
    -   `output_hidden_states=True` returns *all* intermediate hidden states; selecting specific layers is not directly supported.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with MobileNetV1.

<PipelineTag pipeline="image-classification"/>

- [`MobileNetV1ForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- See also: [Image classification task guide](../tasks/image_classification)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

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
