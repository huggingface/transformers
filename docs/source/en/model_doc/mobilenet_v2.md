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

MobileNet V2 builds upon MobileNet V1, improving performance on mobile and embedded devices. Its key innovations are the **inverted residual blocks** (where shortcuts connect thin bottleneck layers) and the use of **linear bottlenecks** (removing non-linearities in narrow layers to preserve information). Like V1, it uses depthwise separable convolutions for efficiency and offers tunable hyperparameters.

You can find checkpoints like [`google/mobilenet_v2_1.4_224`](https://huggingface.co/google/mobilenet_v2_1.4_224) (for classification) or [`google/deeplabv3_mobilenet_v2_1.0_513`](https://huggingface.co/google/deeplabv3_mobilenet_v2_1.0_513) (for segmentation) on the Hub. Check the [Google organization page](https://huggingface.co/google) for other variants. <!-- Consider linking to a dedicated Collection if one exists -->

> [!TIP]
> Click on the MobileNet V2 models in the right sidebar for more examples of how to apply MobileNet to different vision tasks.


The examples below demonstrate how to use MobileNetV2 for image classification and semantic segmentation with [`Pipeline`] or [`AutoModel`].

**Image Classification**

<hfoptions id="usage-img-class">
<hfoption id="Pipeline">

```python
import requests
from PIL import Image
from transformers import pipeline

# Use a pipeline as a high-level helper
pipe = pipeline("image-classification", model="google/mobilenet_v2_1.4_224") # Example checkpoint

# Load image from the web
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Pass image to the pipeline
results = pipe(image)
print(results)
# Example output: [{'label': 'tabby, tabby cat', 'score': 0.918...}, ...]
```

</hfoption>
<hfoption id="AutoModel">

```python
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load processor and model
processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.4_224") # Example checkpoint
model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.4_224")

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

**Semantic Segmentation**

<hfoptions id="usage-sem-seg">
<hfoption id="Pipeline">

```python
import requests
from PIL import Image
from transformers import pipeline

# Use a pipeline as a high-level helper
pipe = pipeline("image-segmentation", model="google/deeplabv3_mobilenet_v2_1.0_513") # Example checkpoint

# Load image from the web
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Pass image to the pipeline
results = pipe(image)
# `results` is a list of masks, one per detected class
# Example showing one mask: print(results[0]['mask'])
# Example output: <PIL.Image.Image image mode=L size=640x480 at 0x...>
```

</hfoption>
<hfoption id="AutoModel">

```python
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# Load processor and model
processor = AutoImageProcessor.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513") # Example checkpoint
model = AutoModelForSemanticSegmentation.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513")

# Load image from the web
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Preprocess image
inputs = processor(images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits # shape (batch_size, num_labels, height, width)

# Post-process to get segmentation map
# Upsample logits to original image size
upsampled_logits = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1], # (height, width)
    mode='bilinear',
    align_corners=False
)
# Get predicted segmentation map
pred_seg = upsampled_logits.argmax(dim=1)[0]
print(pred_seg.shape)
# Example output: torch.Size([480, 640])
```

</hfoption>
</hfoptions>

<!-- Quantization - Not applicable -->
<!-- Attention Visualization - Not applicable for this model type -->

## Notes

-   **Checkpoint Naming:** Classification checkpoints often follow `mobilenet_v2_{depth_multiplier}_{resolution}`, like `mobilenet_v2_1.4_224`. Segmentation checkpoints (using DeepLabV3+ head) might have names like `deeplabv3_mobilenet_v2_{depth_multiplier}_{resolution}`.
-   **Variable Input Size:** Like V1, the model works with images of different sizes (minimum 32x32), handled by [`MobileNetV2ImageProcessor`].
-   **1001 Classes (Classification):** ImageNet-1k pretrained classification models output 1001 classes (index 0 is background).
-   **Segmentation Head:** Segmentation models use a [DeepLabV3+](https://arxiv.org/abs/1802.02611) head, often pretrained on datasets like PASCAL VOC.
-   **Padding Differences:** Similar to V1, original TensorFlow checkpoints had dynamic padding. The HF PyTorch implementation uses static padding by default. Enable dynamic padding (TF behavior) via `tf_padding=True` in [`MobileNetV2Config`].
    ```python
    from transformers import MobileNetV2Config

    # Example: Load config with dynamic padding enabled
    config = MobileNetV2Config.from_pretrained("google/mobilenet_v2_1.4_224", tf_padding=True)
    ```
-   **Unsupported Features:**
    -   The HF implementation uses global average pooling, not the optional fixed 7x7 average pooling from the original paper.
    -   Extracting specific intermediate hidden states (e.g., from expansion layers 10/13) requires `output_hidden_states=True` (returning all states).
    -   For segmentation models, the final convolution layer of the backbone is computed even though the DeepLabV3+ head doesn't use it.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with MobileNetV2.

<PipelineTag pipeline="image-classification"/> <PipelineTag pipeline="image-segmentation"/>

-   **Image Classification:**
    -   [`MobileNetV2ForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
    -   See also: [Image classification task guide](../tasks/image_classification)
-   **Semantic Segmentation:**
    -   See also: [Semantic segmentation task guide](../tasks/semantic_segmentation)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## MobileNetV2Config

[[autodoc]] MobileNetV2Config

## MobileNetV2FeatureExtractor

[[autodoc]] MobileNetV2FeatureExtractor
    - preprocess
    - post_process_semantic_segmentation

## MobileNetV2ImageProcessor

[[autodoc]] MobileNetV2ImageProcessor
    - preprocess

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
