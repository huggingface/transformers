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

# 自定义层和工具

此页面列出了库使用的所有自定义层，以及它为模型提供的实用函数。

其中大多数只有在您研究库中模型的代码时才有用。


## Pytorch自定义模块

[[autodoc]] pytorch_utils.Conv1D

## PyTorch帮助函数

[[autodoc]] pytorch_utils.apply_chunking_to_forward

[[autodoc]] pytorch_utils.find_pruneable_heads_and_indices

[[autodoc]] pytorch_utils.prune_layer

[[autodoc]] pytorch_utils.prune_conv1d_layer

[[autodoc]] pytorch_utils.prune_linear_layer

## TensorFlow自定义层

[[autodoc]] modeling_tf_utils.TFConv1D

[[autodoc]] modeling_tf_utils.TFSequenceSummary

## TensorFlow loss 函数

[[autodoc]] modeling_tf_utils.TFCausalLanguageModelingLoss

[[autodoc]] modeling_tf_utils.TFMaskedLanguageModelingLoss

[[autodoc]] modeling_tf_utils.TFMultipleChoiceLoss

[[autodoc]] modeling_tf_utils.TFQuestionAnsweringLoss

[[autodoc]] modeling_tf_utils.TFSequenceClassificationLoss

[[autodoc]] modeling_tf_utils.TFTokenClassificationLoss

## TensorFlow帮助函数

[[autodoc]] modeling_tf_utils.get_initializer

[[autodoc]] modeling_tf_utils.keras_serializable

[[autodoc]] modeling_tf_utils.shape_list
