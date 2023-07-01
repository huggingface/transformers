<!--版权所有 2020 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非符合许可证的规定，否则您不得使用此文件。您可以在以下网址获取许可证副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按“原样”分发的，无任何明示或暗示的保证或条件。请参阅许可证以了解特定语言下的权限和限制。⚠️ 请注意，此文件是 Markdown 格式的，但包含我们的文档生成器的特定语法（类似于 MDX），可能无法在 Markdown 阅读器中正确渲染。
⚠️ 请注意，此文件是 Markdown 格式的，但包含我们的文档生成器的特定语法（类似于 MDX），可能无法在 Markdown 阅读器中正确渲染。⚠️ 请注意，此文件是 Markdown 格式的，但包含我们的文档生成器的特定语法（类似于 MDX），可能无法在 Markdown 阅读器中正确渲染。
-->
# FSMT

**免责声明：** 如果您发现任何异常情况，请提交 [Github 问题](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title) 并指派给@stas00。

## 概述

FSMT（FairSeq MachineTranslation）模型是由 Nathan Ng，Kyra Yee，Alexei Baevski，Myle Ott，Michael Auli，Sergey Edunov 在 [Facebook FAIR 的 WMT19 新闻翻译任务提交](https://arxiv.org/abs/1907.06616) 中引入的。

论文的摘要如下：

*本文描述了 Facebook FAIR 对 WMT19 共享新闻翻译任务的提交。我们参与了两个语言对和四个语言方向，英语 <-> 德语和英语 <-> 俄语。继续我们去年的提交，我们的基线系统是使用 Fairseq 序列建模工具包训练的大型 BPE-based transformer 模型，依赖于采样的反向翻译。今年，我们尝试了不同的双文本数据过滤方案，以及添加了经过过滤的反向翻译数据。我们还使用领域特定数据对模型进行了集成和微调，然后使用噪声通道模型重新排序进行解码。我们的提交在所有四个方向的人工评估活动中排名第一。在 En-> De 方向上，我们的系统明显优于其他系统和人工翻译。该系统在 BLEU 得分上比我们的 WMT'18 提交提高了 4.5 个点。* 


此模型由 [stas](https://huggingface.co/stas) 贡献。原始代码可以在此处找到 [here](https://github.com/pytorch/fairseq/tree/master/examples/wmt19)。

## 实施备注

- FSMT 使用的源词汇和目标词汇对没有合并到一起。它也不共享嵌入标记。其标记器与 [`XLMTokenizer`] 非常相似，主模型派生自  [`BartModel`]。  [`BartModel`].


## FSMTConfig

[[autodoc]] FSMTConfig

## FSMTTokenizer

[[autodoc]] FSMTTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## FSMTModel

[[autodoc]] FSMTModel
    - forward

## FSMTForConditionalGeneration

[[autodoc]] FSMTForConditionalGeneration
    - forward
