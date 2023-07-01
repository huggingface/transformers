<!--版权所有 2020 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合许可证的要求，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按原样分发的，不附带任何明示或暗示的担保或条件。请查看许可证以了解特定语言下的权限和限制。⚠️ 注意，此文件为 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确显示
-->



# Custom Layers and Utilities

本页面列出了库中使用的所有自定义层，以及用于建模的实用函数。
大多数情况下，这些仅在研究库中模型的代码时才有用。

## PyTorch 自定义模块
[[autodoc]] pytorch_utils.Conv1D
[[autodoc]] modeling_utils.PoolerStartLogits    - forward
[[autodoc]] modeling_utils.PoolerEndLogits    - forward
[[autodoc]] modeling_utils.PoolerAnswerClass    - forward
[[autodoc]] modeling_utils.SquadHeadOutput
[[autodoc]] modeling_utils.SQuADHead    - forward
[[autodoc]] modeling_utils.SequenceSummary    - forward

## PyTorch 辅助函数

[[autodoc]] pytorch_utils.apply_chunking_to_forward
[[autodoc]] pytorch_utils.find_pruneable_heads_and_indices
[[autodoc]] pytorch_utils.prune_layer
[[autodoc]] pytorch_utils.prune_conv1d_layer
[[autodoc]] pytorch_utils.prune_linear_layer
## TensorFlow 自定义层
[[autodoc]] modeling_tf_utils.TFConv1D
[[autodoc]] modeling_tf_utils.TFSequenceSummary
## TensorFlow 损失函数
[[autodoc]] modeling_tf_utils.TFCausalLanguageModelingLoss
[[autodoc]] modeling_tf_utils.TFMaskedLanguageModelingLoss
[[autodoc]] modeling_tf_utils.TFMultipleChoiceLoss
[[autodoc]] modeling_tf_utils.TFQuestionAnsweringLoss
[[autodoc]] modeling_tf_utils.TFSequenceClassificationLoss
[[autodoc]] modeling_tf_utils.TFTokenClassificationLoss

## TensorFlow 辅助函数
[[autodoc]] modeling_tf_utils.get_initializer
[[autodoc]] modeling_tf_utils.keras_serializable
[[autodoc]] modeling_tf_utils.shape_list