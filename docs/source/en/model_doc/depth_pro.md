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
*This model was released on 2024-10-02 and added to Hugging Face Transformers on 2025-02-10 and contributed by [geetu040](https://github.com/geetu040).*

# DepthPro

[DepthPro](https://huggingface.co/papers/2410.02073) is a foundation model for zero-shot metric monocular depth estimation, generating high-resolution depth maps with sharpness and fine details. It uses a multi-scale Vision Transformer (ViT)-based architecture with a shared Dinov2 encoder and a DPT-like fusion stage for precise depth estimation. The model achieves metric accuracy without camera intrinsics and produces a 2.25-megapixel depth map in 0.3 seconds on a standard GPU. Technical contributions include an efficient multi-scale vision transformer, a combined real and synthetic dataset training protocol, and state-of-the-art focal length estimation from a single image.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="depth-estimation", model="apple/DepthPro-hf", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
import requests
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

image_processor = AutoImageProcessor.from_pretrained("apple/DepthPro-hf")
model = AutoModelForDepthEstimation.from_pretrained("apple/DepthPro-hf", dtype="auto")
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

post_processed_output = image_processor.post_process_depth_estimation(
    outputs,
    target_sizes=[(image.height, image.width)],
)
field_of_view = post_processed_output[0]["field_of_view"]
focal_length = post_processed_output[0]["focal_length"]
predicted_depth = post_processed_output[0]["predicted_depth"]
depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
depth = depth.detach().cpu().numpy() * 255
Image.fromarray(depth.astype("uint8"))
```

</hfoption>
</hfoptions>

## Usage tips

- The DepthPro model processes input images by downsampling at multiple scales and splitting each scaled version into patches. These patches encode using a shared Vision Transformer (ViT)-based Dinov2 patch encoder. The full image processes through a separate image encoder.
- Extracted patch features merge into feature maps, upsample, and fuse using a DPT-like decoder to generate the final depth estimation. If enabled, an additional Field of View (FOV) encoder processes the image for estimating the camera's field of view, aiding in depth accuracy.
- [`DepthProForDepthEstimation`] uses a [`DepthProEncoder`] for encoding the input image and a [`FeatureFusionStage`] for fusing output features from the encoder.
- The [`DepthProEncoder`] uses two encoders:

    - patch_encoder: Input image scales with multiple ratios as specified in the `scaled_images_ratios` configuration. Each scaled image splits into smaller patches of size `patch_size` with overlapping areas determined by `scaled_images_overlap_ratios`. These patches process through the patch_encoder.
    - image_encoder: Input image rescales to `patch_size` and processes through the image_encoder.

- Both encoders configure via `patch_model_config` and `image_model_config` respectively. Both default to separate [`Dinov2Model`] instances.
- Outputs from both encoders (`last_hidden_state`) and selected intermediate states (`hidden_states`) from patch_encoder fuse by a DPT-based [`FeatureFusionStage`] for depth estimation.
- The network supplements with a focal length estimation head. A small convolutional head ingests frozen features from the depth estimation network and task-specific features from a separate ViT image encoder to predict the horizontal angular field-of-view.
- The `use_fov_model` parameter in [`DepthProConfig`] controls whether FOV prediction is enabled. By default, it's set to `False` to conserve memory and computation.
- When enabled, the FOV encoder instantiates based on the `fov_model_config` parameter, which defaults to a [`Dinov2Model`]. The `use_fov_model` parameter also passes when initializing the [`DepthProForDepthEstimation`] model.
- The pretrained model at checkpoint `apple/DepthPro-hf` uses the FOV encoder. Set `use_fov_model=False` when loading the model to use the pretrained model without FOV encoder, which saves computation.
- To instantiate a new model with FOV encoder, set `use_fov_model=True` in the config. Or set `use_fov_model=True` when initializing the model, which overrides the value in config.

## DepthProConfig

[[autodoc]] DepthProConfig

## DepthProImageProcessor

[[autodoc]] DepthProImageProcessor
    - preprocess
    - post_process_depth_estimation

## DepthProImageProcessorFast

[[autodoc]] DepthProImageProcessorFast
    - preprocess
    - post_process_depth_estimation

## DepthProModel

[[autodoc]] DepthProModel
    - forward

## DepthProForDepthEstimation

[[autodoc]] DepthProForDepthEstimation
    - forward

