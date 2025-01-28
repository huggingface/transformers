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

# DepthPro

## Overview

The DepthPro model was proposed in [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/abs/2410.02073) by Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, Vladlen Koltun.

It leverages a multi-scale [Vision Transformer (ViT)](vit) optimized for dense predictions. It downsamples an image at several scales. At each scale, it is split into patches, which are processed by a ViT-based [Dinov2](dinov2) patch encoder, with weights shared across scales. Patches are merged into feature maps, upsampled, and fused via a [DPT](dpt)-like decoder.

The abstract from the paper is the following:

*We present a foundation model for zero-shot metric monocular depth estimation. Our model, Depth Pro, synthesizes high-resolution depth maps with unparalleled sharpness and high-frequency details. The predictions are metric, with absolute scale, without relying on the availability of metadata such as camera intrinsics. And the model is fast, producing a 2.25-megapixel depth map in 0.3 seconds on a standard GPU. These characteristics are enabled by a number of technical contributions, including an efficient multi-scale vision transformer for dense prediction, a training protocol that combines real and synthetic datasets to achieve high metric accuracy alongside fine boundary tracing, dedicated evaluation metrics for boundary accuracy in estimated depth maps, and state-of-the-art focal length estimation from a single image. Extensive experiments analyze specific design choices and demonstrate that Depth Pro outperforms prior work along multiple dimensions.*

<img src="https://huggingface.co/geetu040/DepthPro/resolve/main/assets/architecture.jpg"
alt="drawing" width="600"/>

<small> DepthPro architecture. Taken from the <a href="https://arxiv.org/abs/2410.02073" target="_blank">original paper</a>. </small>

This model was contributed by [geetu040](https://github.com/geetu040). The original code can be found [here](https://github.com/apple/ml-depth-pro).

<!-- TODO -->

Here's an improved version of your documentation with enhanced clarity, formatting, and structure for easier understanding:

---

## **Usage Tips**

Initialize the Model with Default Configuration
```python
from transformers import DepthProConfig, DepthProModel

config = DepthProConfig()
model = DepthProModel(config=config)
```

Load a Pre-Trained Model for Depth Estimation
```python
from transformers import DepthProConfig, DepthProForDepthEstimation

checkpoint = "geetu040/DepthPro"
model = DepthProForDepthEstimation.from_pretrained(checkpoint)
config = model.config
```

Key Features and Configuration Details

1. Dual-Encoder Architecture:
   - The `DepthProModel` uses **two encoders**:
     - **`image_encoder`** and **`patch_encoder`**, which can be configured via `image_model_config` and `patch_model_config` in the configuration.
   - By default, and in the pre-trained model, both encoders use the **`Dinov2Model`** architecture.

2. Image Scaling and Patch Processing:
   - Input images are scaled with multiple ratios, as specified in the `scaled_images_ratios` configuration.
   - Each scaled image is split into smaller **patches** of size `patch_size` with overlapping areas determined by `scaled_images_overlap_ratios`.
   - These patches are processed by the **`patch_encoder`**, while the image is also rescaled to `patch_size` and is processed by the **`image_encoder`**.
   - Outputs from both encoders (`last_hidden_state`) and selected intermediate states (`hidden_states`) from **`patch_encoder`** are fused by a `DPT`-based `FeatureFusionStage` for depth estimation.

3. Optional Field of View (FOV) Prediction:
   - If `use_fov_model` is set to `True` in the configuration, the model predicts the **Field of View (FOV)** using a third encoder.
   - This encoder also scales the image to `patch_size` and uses its `last_hidden_state` for FOV prediction. The encoder can be specified in the configuration using `fov_model_config`.

4. Configuration and Validation:
   - All encoders receive input images of size `patch_size`.
   - The `image_size` for each encoder in the configuration should match the `patch_size`. This is validated when creating a `DepthProConfig`.

5. Preprocessing and Postprocessing:
   - Use the `DepthProImageProcessor` for preparing inputs and processing outputs:
     - **Preprocessing**: Prepare images (rescale, normalize, resize) for model input.
     - **Postprocessing**: Use `DepthProImageProcessor.post_process_depth_estimation` to interpolate the predicted depth to match the original input image size.

6. Support for Variable Resolution and Aspect Ratios:
   - The `DepthProModel` can process images with different resolutions and aspect ratios. However, for generating predicted depths that match the input image size, ensure the configuration satisfies:
   ```py
   input_image_size / 2**(n_fusion_blocks + 1) == image_model_config.image_size / image_model_config.patch_size
   ```

   - **Where**:
     - `input_image_size`: The size of the input image.
     - `image_model_config.image_size`: Image size for **`image_encoder`** which equals to `patch_size` in `DepthProConfig`.
     - `n_fusion_blocks`: Total fusion blocks, calculated as:
       ```py
       len(intermediate_hook_ids) + len(scaled_images_ratios)
       ```

### **Customizing Encoders in `DepthProModel`**

The `DepthProModel` architecture uses **three encoders**, each responsible for a specific task:

1. **Patch Encoder**: Processes image patches created by splitting the input image.
2. **Image Encoder**: Processes the input image resized to `patch_size`.
3. **FOV (Field of View) Encoder**: Generates the Field of View (FOV), if `use_fov_model` is enabled.

You can configure each encoder to use any compatible model architecture. For example, to use:
- **`ViT` (Vision Transformer)** as the **patch encoder**, and
- **`BEiT`** as the **image encoder**, and
- **`DinoV2`** as the **FOV encoder**.

```python
from transformers import DepthProConfig, DepthProForDepthEstimation

config = DepthProConfig(
    patch_model_config={
        "model_type": "vit",
        "num_hidden_layers": 6,
        "patch_size": 16,
        "hidden_size": 512,
        "num_attention_heads": 16,
        "image_size": 384,  # matches `patch_size`
    },
    image_model_config={
        "model_type": "beit",
        "num_hidden_layers": 4,
        "patch_size": 8,
        "hidden_size": 256,
        "num_attention_heads": 8,
        "image_size": 384,  # matches `patch_size`
    },
    fov_model_config={
        "model_type": "dinov2",
        "num_hidden_layers": 4,
        "patch_size": 8,
        "hidden_size": 256,
        "num_attention_heads": 8,
        "image_size": 384,  # matches `patch_size`
    },
    patch_size=384,
    # uses layers from the patch encoder
    intermediate_hook_ids=[5, 1],
    use_fov_model=True,
)
model = DepthProForDepthEstimation(config)
```

### Using Scaled Dot Product Attention (SDPA)

PyTorch includes a native scaled dot-product attention (SDPA) operator as part of `torch.nn.functional`. This function 
encompasses several implementations that can be applied depending on the inputs and the hardware in use. See the 
[official documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 
or the [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)
page for more information.

SDPA is used by default for `torch>=2.1.1` when an implementation is available, but you may also set 
`attn_implementation="sdpa"` in `from_pretrained()` to explicitly request SDPA to be used.

```py
from transformers import DepthProForDepthEstimation
model = DepthProForDepthEstimation.from_pretrained("geetu040/DepthPro", attn_implementation="sdpa", torch_dtype=torch.float16)
```

For the best speedups, we recommend loading the model in half-precision (e.g. `torch.float16` or `torch.bfloat16`).

On a local benchmark (A100-40GB, PyTorch 2.3.0, OS Ubuntu 22.04) with `float32` and `google/vit-base-patch16-224` model, we saw the following speedups during inference.

|   Batch size |   Average inference time (ms), eager mode |   Average inference time (ms), sdpa model |   Speed up, Sdpa / Eager (x) |
|--------------|-------------------------------------------|-------------------------------------------|------------------------------|
|            1 |                                         7 |                                         6 |                      1.17 |
|            2 |                                         8 |                                         6 |                      1.33 |
|            4 |                                         8 |                                         6 |                      1.33 |
|            8 |                                         8 |                                         6 |                      1.33 |

## Resources

- Research Paper: [Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/pdf/2410.02073)
- Official Implementation: [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro)
- DepthPro for Super Resolution and Image Segmentation
    - Read blog on Medium: [Depth Pro: Beyond Depth](https://medium.com/@raoarmaghanshakir040/depth-pro-beyond-depth-9d822fc557ba)
    - Code on Github: [geetu040/depthpro-beyond-depth](https://github.com/geetu040/depthpro-beyond-depth)

<!-- A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with DepthPro. -->

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

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
