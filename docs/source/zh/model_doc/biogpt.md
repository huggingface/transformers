<!--版权所有2022年The HuggingFace团队保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非遵守许可证，否则不得使用此文件。您可以在以下网址获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的担保或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。
-->
# BioGPT

## 概述

BioGPT 模型是由 Renqian Luo、Liai Sun、Yingce Xia、Tao Qin、Sheng Zhang、Hoifung Poon 和 Tie-Yan Liu 提出的 [BioGPT：面向生物医学文本生成和挖掘的生成预训练 Transformer](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbac409/6713511?guestAccessKey=a66d9b5d-4f83-4017-bb52-405815c907b9)。BioGPT 是一个面向领域的生成预训练 Transformer 语言模型，用于生物医学文本生成和挖掘。BioGPT 遵循 Transformer 语言模型的主干，并从头开始对 1500 万篇 PubMed 摘要进行预训练。

下面是该论文的摘要：
*预训练语言模型在生物医学领域受到越来越多的关注，受到其在通用自然语言领域取得的巨大成功的启发。在通用语言领域的两个主要预训练语言模型分支中，即 BERT（及其变体）和 GPT（及其变体），第一个已经在生物医学领域进行了广泛研究，例如 BioBERT 和 PubMedBERT。虽然它们在各种鉴别性生物医学任务上取得了巨大成功，但生成能力的缺乏限制了它们的应用范围。在本文中，我们提出了 BioGPT，一种面向领域的生成 Transformer 语言模型，它在大规模生物医学文献中进行了预训练。我们在六个生物医学自然语言处理任务上评估了 BioGPT，并证明我们的模型在大多数任务上优于先前的模型。尤其是，我们在 BC5CDR、KD-DTI 和 DDI 端到端关系抽取任务上分别获得了 44.98%、38.42%和 40.76%的 F1 分数，并在 PubMedQA 上获得了 78.2%的准确率，创造了新纪录。我们对文本生成的案例研究进一步证明了 BioGPT 在生物医学文献中生成流利描述的优势。*



提示：

- BioGPT 是一个具有绝对位置嵌入的模型，因此通常建议在右侧而不是左侧填充输入。- BioGPT 是使用因果语言建模（CLM）目标进行训练的，因此在预测序列中的下一个标记时非常强大。利用这个特性，BioGPT 能够生成语法连贯的文本，就像在 run_generation.py 示例脚本中可以观察到的那样。- 模型可以接受 `past_key_values`（对于 PyTorch 而言）作为输入，这是先前计算的键/值注意力对。使用此（past_key_values 或 past）值可防止模型在文本生成的上下文中重新计算预计算的值。有关其用法的更多信息，请参阅 BioGptForCausalLM.forward()方法的 past_key_values 参数。

此模型由 [kamalkraj](https://huggingface.co/kamalkraj) 贡献。原始代码可以在 [此处](https://github.com/microsoft/BioGPT) 找到。

## 文档资源

- [因果语言建模任务指南](../tasks/language_modeling)

## BioGptConfig
[[autodoc]] BioGptConfig

## BioGptTokenizer

[[autodoc]] BioGptTokenizer
    - save_vocabulary


## BioGptModel

[[autodoc]] BioGptModel
    - forward


## BioGptForCausalLM

[[autodoc]] BioGptForCausalLM
    - forward

    
## BioGptForTokenClassification

[[autodoc]] BioGptForTokenClassification
    - forward


## BioGptForSequenceClassification

[[autodoc]] BioGptForSequenceClassification
    - forward