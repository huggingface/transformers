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

# SmolVLM

## Overview
SmolVLM2 is an adaptation of the Idefics3 model with three main differences:

- It uses SmolLM2 for the text model.
- It supports multi-image and video inputs

## Usage tips

Input images are processed either by upsampling (if resizing is enabled) or at their original resolution. The resizing behavior depends on two parameters: do_resize and size.

Videos should not be upsampled. 

If `do_resize` is set to `True`, the model resizes images so that the longest edge is 4*364 pixels by default.
The default resizing behavior can be customized by passing a dictionary to the `size` parameter. For example, `{"longest_edge": 4 * 364}` is the default, but you can change it to a different value if needed.

Here’s how to control resizing and set a custom size:
```python
image_processor = SmolVLMImageProcessor(do_resize=True, size={"longest_edge": 2 * 364}, max_image_size=364)
```

Additionally, the `max_image_size` parameter, which controls the size of each square patch the image is decomposed into, is set to 364 by default but can be adjusted as needed. After resizing (if applicable), the image processor decomposes the images into square patches based on the `max_image_size` parameter.

This model was contributed by [orrzohar](https://huggingface.co/orrzohar).


## SmolVLMConfig

[[autodoc]] SmolVLMConfig

## SmolVLMVisionConfig

[[autodoc]] SmolVLMVisionConfig

## Idefics3VisionTransformer

[[autodoc]] SmolVLMVisionTransformer

## SmolVLMModel

[[autodoc]] SmolVLMModel
    - forward

## SmolVLMForConditionalGeneration

[[autodoc]] SmolVLMForConditionalGeneration
    - forward


## SmolVLMImageProcessor
[[autodoc]] SmolVLMImageProcessor
    - preprocess


## SmolVLMProcessor
[[autodoc]] SmolVLMProcessor
    - __call__
