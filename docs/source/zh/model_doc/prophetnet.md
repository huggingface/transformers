<!--版权所有 2020 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证 2.0 版（“许可证”）许可；除非符合许可证要求，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按“原样”分发，不附带任何明示或暗示的担保或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法正常在您的 Markdown 查看器中呈现。
-->
# ProphetNet

<div class="flex flex-wrap space-x-1"> <a href="https://huggingface.co/models?filter=prophetnet"> <img alt="Models" src="https://img.shields.io/badge/All_model_pages-prophetnet-blueviolet"> </a> <a href="https://huggingface.co/spaces/docs-demos/prophetnet-large-uncased"> <img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"> </a> </div>

**免责声明：** 如果您发现任何问题，请提出 [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title) 并指定@patrickvonplaten

## 概览

ProphetNet 模型是由 Yu Yan、Weizhen Qi、Yeyun Gong、Dayiheng Liu、Nan Duan、Jiusheng Chen、RuofeiZhang 和 Ming Zhou 于 2020 年 1 月 13 日提出的 [ProphetNet：用于序列到序列预训练的未来 N-gram 预测](https://arxiv.org/abs/2001.04063)。
ProphetNet 是一种编码器-解码器模型，可以对“ngram”语言建模进行 n 个未来标记的预测，而不仅仅是下一个标记。

论文中的摘要如下所示：

*在本文中，我们提出了一种名为 ProphetNet 的新型序列到序列预训练模型，该模型引入了一种新颖的自监督目标，称为未来 n-gram 预测，以及所提出的 n 流自注意机制。ProphetNet 通过对传统序列到序列模型中的一步预测进行优化，通过在每个时间步根据先前上下文标记同时预测下一个 n 个标记来进行 n 步预测。未来的 n-gram 预测明确鼓励模型为未来的标记进行规划，并防止对强局部相关性进行过拟合。我们使用基准规模数据集（16GB）和大规模数据集（160GB）对 ProphetNet 进行预训练。然后，我们对 CNN/DailyMail、Gigaword 和 SQuAD 1.1 进行实验。 进行摘要生成和问题生成任务。实验结果表明，与使用相同规模预训练语料库的模型相比，ProphetNet 在所有这些数据集上都取得了新的最先进的结果。*

提示：

- ProphetNet 是一个带有绝对位置嵌入的模型，因此通常建议在右侧而不是左侧进行填充输入。  
- 模型架构基于原始 Transformer，但通过主要自注意机制和自流和 n 流（预测）自注意机制替换了解码器中的“标准”自注意机制。
作者的代码可以在 [此处](https://github.com/microsoft/ProphetNet) 找到。

## 文档资源

- [因果语言建模任务指南](../tasks/language_modeling)
- [翻译任务指南](../tasks/translation)
- [摘要生成任务指南](../tasks/summarization)

## ProphetNetConfig

[[autodoc]] ProphetNetConfig

## ProphetNetTokenizer

[[autodoc]] ProphetNetTokenizer

## ProphetNet specific outputs

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput

## ProphetNetModel

[[autodoc]] ProphetNetModel
    - forward

## ProphetNetEncoder

[[autodoc]] ProphetNetEncoder
    - forward

## ProphetNetDecoder

[[autodoc]] ProphetNetDecoder
    - forward

## ProphetNetForConditionalGeneration

[[autodoc]] ProphetNetForConditionalGeneration
    - forward

## ProphetNetForCausalLM

[[autodoc]] ProphetNetForCausalLM
    - forward
