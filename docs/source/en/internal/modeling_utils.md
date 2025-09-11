<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Custom Layers and Utilities

This page lists all the custom layers used by the library, as well as the utility functions and classes it provides for modeling.

Most of those are only useful if you are studying the code of the models in the library.

## Layers

[[autodoc]] GradientCheckpointingLayer

## Attention Functions

[[autodoc]] AttentionInterface
    - register

## Attention Mask Functions

[[autodoc]] AttentionMaskInterface
    - register

## Rotary Position Embedding Functions

[[autodoc]] dynamic_rope_update

## Pytorch custom modules

[[autodoc]] pytorch_utils.Conv1D

## PyTorch Helper Functions

[[autodoc]] pytorch_utils.apply_chunking_to_forward

[[autodoc]] pytorch_utils.find_pruneable_heads_and_indices

[[autodoc]] pytorch_utils.prune_layer

[[autodoc]] pytorch_utils.prune_conv1d_layer

[[autodoc]] pytorch_utils.prune_linear_layer
