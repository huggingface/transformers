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

# Image Processor

An image processor is in charge of loading images (optionally), preparing input features for vision models and post processing their outputs. This includes transformations such as resizing, normalization, and conversion to PyTorch and Numpy tensors. It may also include model specific post-processing such as converting logits to segmentation masks.

Image processors use a backend-based architecture. The class hierarchy is:

- [`BaseImageProcessor`] — abstract base class (for backward compatibility only; do not instantiate directly)
  - [`TorchvisionBackend`] — the **default** [torchvision-backed](https://pytorch.org/vision/stable/index.html) backend. GPU-accelerated and significantly faster than the PIL backend. All models expose a `<Model>ImageProcessor` class that inherits from it.
  - [`PilBackend`] — the PIL/NumPy alternative backend. Portable, CPU-only. Only available for older models via a `<Model>ImageProcessorPil` class; useful when exact numerical parity with the original implementation is required.

Both backends expose the same API. Use the `backend` attribute to inspect which backend a loaded processor uses (e.g. `processor.backend == "torchvision"`).

Use [`AutoImageProcessor.from_pretrained`] with the `backend` argument to select a backend. When `backend` is omitted (the default), torchvision is picked when it is installed and PIL is used otherwise. Pass an explicit string to override that choice:

```python
from transformers import AutoImageProcessor

# Default: picks torchvision if available, otherwise pil
processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Explicitly request torchvision
processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50", backend="torchvision")

# Explicitly request PIL
processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50", backend="pil")
```

When using the torchvision backend, you can set the `device` argument to specify the device on which the processing should be done. By default, the processing is done on the same device as the inputs if the inputs are tensors, or on the CPU otherwise.

```python
from torchvision.io import read_image
from transformers import DetrImageProcessor

images = read_image("image.jpg")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
images_processed = processor(images, return_tensors="pt", device="cuda")
```

Here are some speed comparisons between the torchvision and PIL backends for the `DETR` and `RT-DETR` models, and how they impact overall inference time:

<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_detr_fast_padded.png" />
</div>
<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_detr_fast_batched_compiled.png" />
</div>

<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_rt_detr_fast_single.png" />
</div>
<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_rt_detr_fast_batched.png" />
</div>

These benchmarks were run on an [AWS EC2 g5.2xlarge instance](https://aws.amazon.com/ec2/instance-types/g5/), utilizing an NVIDIA A10G Tensor Core GPU.

## ImageProcessingMixin

[[autodoc]] image_processing_utils.ImageProcessingMixin
    - from_pretrained
    - save_pretrained

## BatchFeature

[[autodoc]] BatchFeature

## BaseImageProcessor

[[autodoc]] image_processing_utils.BaseImageProcessor

## TorchvisionBackend

[[autodoc]] image_processing_backends.TorchvisionBackend

## PilBackend

[[autodoc]] image_processing_backends.PilBackend
