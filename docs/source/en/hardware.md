<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Hardware

This section brings together hardware setup and device-specific configuration for training and inference. Start with the [Installation](./installation#python) guide to install Transformers and PyTorch for your device, then follow the relevant hardware guide.

| Hardware | Guide |
|---|---|
| CPU | [CPU training](./perf_train_cpu), including mixed precision and distributed training. |
| NVIDIA GPU | [Installation](./installation#python) and [GPU workstation setup](./perf_hardware), including power, cooling, and multi-GPU connectivity. |
| Intel GPU | [Installation](./installation#python) and [accelerator selection](./accelerator_selection), including device visibility and ordering. |
| Apple Silicon | [MPS configuration](./perf_train_special), including model loading, mixed precision, and graph cache management. |
| Intel Gaudi | [HPU configuration](./perf_train_gaudi), including execution modes, mixed precision, compilation, and distributed training. |

For techniques that can be used across hardware, see the [Optimization](./optimization_overview) overview. Support for each technique depends on the device and backend.
