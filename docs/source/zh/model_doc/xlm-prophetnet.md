<!--版权所有2020年HuggingFace团队。保留所有权利。-->
根据 Apache 许可证第 2 版（“许可证”）获得许可，您不得使用此文件，除非符合许可证的要求。您可以在以下网址获得许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按原样分发的，不附带任何明示或暗示的担保或条件。有关许可证下的特定语言的权限和限制，请参阅许可证。an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
特别提示：此文件是 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能在您的 Markdown 查看器中无法正确显示。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能在您的 Markdown 查看器中无法正确显示。请注意，此文件是 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能在您的 Markdown 查看器中无法正确显示。
-->
# XLM-ProphetNet

<div class="flex flex-wrap space-x-1"> <a href="https://huggingface.co/models?filter=xprophetnet"> <img alt="Models" src="https://img.shields.io/badge/All_model_pages-xprophetnet-blueviolet"> </a> <a href="https://huggingface.co/spaces/docs-demos/xprophetnet-large-wiki100-cased-xglue-ntg"> <img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"> </a> </div>

**免责声明：** 如果您发现任何奇怪的地方，请提交 [Github 问题](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title) 并分配给@patrickvonplaten

## 概述

XLM-ProphetNet 模型是由 Yu Yan、Weizhen Qi、Yeyun Gong、Dayiheng Liu、Nan Duan、Jiusheng Chen、Ruofei Zhang 和 Ming Zhou 于 2020 年 1 月 13 日提出的 [《ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training》](https://arxiv.org/abs/2001.04063)。

XLM-ProphetNet 是一个编码器-解码器模型，可以预测“ngram”语言建模的 n 个未来标记，而不仅仅是下一个标记。它的架构与 ProphetNet 相同，但该模型是在多语言的“wiki100”维基百科转储上训练的。

该论文的摘要如下：
*在本文中，我们提出了一种名为 ProphetNet 的新型序列到序列预训练模型，它引入了一种称为未来 n-gram 预测的新颖的自监督目标和提出的 n 流自注意机制。ProphetNet 的优化不是传统序列到序列模型中的一步预测优化，而是基于每个时间步的先前上下文标记预测下一个 n 个标记。未来 n-gram 预测明确地鼓励模型为未来标记制定计划，并防止在强烈的局部相关性上过拟合。我们使用基本规模数据集（16GB）和大规模数据集（160GB）分别对 ProphetNet 进行预训练。然后，我们对 CNN/DailyMail、Gigaword 和 SQuAD 1.1 基准进行抽象摘要和问题生成任务的实验。实验结果表明，与使用相同规模预训练语料库的模型相比，ProphetNet 在所有这些数据集上都取得了最新的最先进的结果。*

作者的代码可以在 [此处](https://github.com/microsoft/ProphetNet) 找到。

提示：

- XLM-ProphetNet 的模型架构和预训练目标与 ProphetNet 相同，但 XLM-ProphetNet 是在跨语言数据集 XGLUE 上进行预训练的。

## 文档资源

- [因果语言建模任务指南](../tasks/language_modeling)
- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)
## XLMProphetNetConfig

[[autodoc]] XLMProphetNetConfig

## XLMProphetNetTokenizer

[[autodoc]] XLMProphetNetTokenizer

## XLMProphetNetModel

[[autodoc]] XLMProphetNetModel

## XLMProphetNetEncoder

[[autodoc]] XLMProphetNetEncoder

## XLMProphetNetDecoder

[[autodoc]] XLMProphetNetDecoder

## XLMProphetNetForConditionalGeneration

[[autodoc]] XLMProphetNetForConditionalGeneration

## XLMProphetNetForCausalLM

[[autodoc]] XLMProphetNetForCausalLM
