<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->


# Video Processor

A **Video Processor** is a utility responsible for preparing input features for Vision Large Language Models (VLMs) that support video data, as well as handling the post-processing of their outputs. It provides transformations such as resizing, normalization, and conversion into PyTorch, TensorFlow, Flax, or NumPy tensors. 

The video processor extends the functionality of image processors by allowing Vision Large Language Models (VLMs) to handle videos with a distinct set of arguments compared to images. It serves as the bridge between raw video data and the model, ensuring that input features are optimized for the VLM.

When adding a new VLM or updating an existing one to enable distinct video preprocessing, saving and reloading the processor configuration will store the video related arguments in a dedicated file named `video_preprocessing_config.json`. Don't worry if you haven't upadted your VLM, the processor will try to load video related configurations from a file named `preprocessing_config.json`.

Currently, the video processor processes video data by treating each frame as an individual image and applying transformations frame-by-frame. While functional, this approach is not highly efficient. Planned updates include the addition of **fast video processors**, leveraging the [torchvision](https://pytorch.org/vision/stable/index.html) library. These updates will introduce GPU acceleration and significantly enhance processing speed, especially for tasks requiring high throughput.

### Usage Example
Here's an example of how to load a video processor with [`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) model:

```python
from transformers import AutoVideoProcessor

processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
```


## BaseVideoProcessor

[[autodoc]] video_processing_utils.BaseVideoProcessor
