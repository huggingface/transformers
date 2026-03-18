<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-04-14 and added to Hugging Face Transformers on 2026-03-13.*

# PPLCNetV3

[PPLCNetV3](https://huggingface.co/papers/2109.15099) is a lightweight CPU-optimized convolutional backbone designed for efficient image classification and downstream vision tasks. It builds on the PP-LCNet architecture with improved training strategies and structural refinements for better accuracy-latency tradeoffs on CPU hardware.

## Notes

- PPLCNetV3 is provided as a backbone network only. No pre-trained image classification checkpoint has been officially released.

## PPLCNetV3Config

[[autodoc]] PPLCNetV3Config

## PPLCNetV3Backbone

[[autodoc]] PPLCNetV3Backbone
