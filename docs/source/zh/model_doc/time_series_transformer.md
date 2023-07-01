<!-- 版权所有2022年HuggingFace团队保留所有权利。
根据Apache许可证第2.0版（"许可证"）的规定，您不得使用此文件，除非符合许可证的要求。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于"原样" BASIS，无论是明示还是暗示，在法律允许的范围内，不提供任何形式的担保或条件。请参阅许可证以获取特定语言下的权限和限制。
⚠️ 请注意，此文件是Markdown格式的，但包含特定于我们的文档生成器（类似于MDX）的语法，您的Markdown查看器可能无法正确显示。
-->

# 时间序列Transformer
<Tip>

这是一个最近推出的模型，因此API尚未经过广泛测试。可能存在一些错误或轻微的变更，以便在将来进行修复。如果您发现任何奇怪的情况，请提交[GitHub Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title)。

</Tip>

## 概述

时间序列Transformer模型是用于时间序列预测的普通编码器-解码器Transformer模型。

提示：

- 在HuggingFace博客中查看关于时间序列Transformer的博文：[使用🤗 Transformers进行概率时间序列预测](https://huggingface.co/blog/time-series-transformers)
- 与库中的其他模型类似，[`TimeSeriesTransformerModel`]是没有任何顶部头的原始Transformer模型，而[`TimeSeriesTransformerForPrediction`]在前者的基础上添加了一个分布头，用于时间序列预测。请注意，这是一种所谓的概率预测模型，而不是点预测模型。这意味着模型学习一个分布，可以从中进行采样，而不是直接输出值。
- [`TimeSeriesTransformerForPrediction`]由两个模块组成：编码器和解码器。编码器以`context_length`个时间序列值作为输入（称为`past_values`），解码器将预测未来`prediction_length`个时间序列值（称为`future_values`）。在训练过程中，需要为模型提供（`past_values`和`future_values`）的配对数据。
- 除了原始的（`past_values`和`future_values`），通常还会为模型提供其他特征。这些特征可以是以下内容：      
- `past_time_features`：模型将将其添加到`past_values`中的时间特征。这些作为Transformer编码器的"位置编码"。    例如，"月份的日期"，"年份的月份"等作为标量值（然后作为向量堆叠在一起）。    例如，如果给定的时间序列值是在8月11日获得的，则可以将[11, 8]作为时间特征向量（其中11代表"日期的日期"，8代表"年份的月份"）。   
- `future_time_features`：模型将将其添加到`future_values`中的时间特征。这些作为Transformer解码器的"位置编码"。    例如，"月份的日期"，"年份的月份"等作为标量值（然后作为向量堆叠在一起）。    例如，如果给定的时间序列值是在8月11日获得的，则可以将[11, 8]作为时间特征向量（其中11代表"日期的日期"，8代表"年份的月份"）。      
- `static_categorical_features`：随时间保持不变的分类特征（即，对于所有`past_values`和`future_values`具有相同的值）。    一个示例是标识给定时间序列的商店ID或区域ID。    请注意，这些特征需要对所有数据点（包括未来的数据点）都已知。   
- `static_real_features`：随时间保持不变的实值特征（即，对于所有`past_values`和`future_values`具有相同的值）。    一个示例是您拥有时间序列值的产品的图像表示（例如，销售鞋子的[ResNet](resnet)嵌入）。    请注意，这些特征需要对所有数据点（包括未来的数据点）都已知。- 模型使用"teacher-forcing"进行训练，类似于Transformer进行机器翻译的训练方式。这意味着在训练过程中，将`future_values`向右移动一个位置作为解码器的输入，前面加上`past_values`的最后一个值。在每个时间步骤中，模型需要预测下一个目标。因此，训练的设置与语言模型的GPT模型类似，只是没有`decoder_start_token_id`（我们只使用上下文的最后一个值作为解码器的初始输入）。- 在推理阶段，我们将`past_values`的最后一个值作为输入传递给解码器。

接下来，我们可以从模型中采样以在下一个时间步骤进行预测，然后将其馈送给解码器以进行下一个预测（也称为自回归生成）。

此模型由[kashif](https://huggingface.co/kashif)贡献。

## TimeSeriesTransformerConfig

[[autodoc]] TimeSeriesTransformerConfig


## TimeSeriesTransformerModel

[[autodoc]] TimeSeriesTransformerModel
    - forward


## TimeSeriesTransformerForPrediction

[[autodoc]] TimeSeriesTransformerForPrediction
    - forward
