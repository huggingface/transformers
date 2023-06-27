<!--版权所有 2020 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证 2.0 版（“许可证”）授权；您除非符合许可证，否则不得使用此文件。您可以在
http://www.apache.org/licenses/LICENSE-2.0
根据适用法律或书面协议要求，按 "按原样" 的方式分发的软件在许可证下分发。对于特定语言的权限和限制，请参阅许可证。
⚠️请注意，此文件是 Markdown 格式的，但包含特定于我们的文档生成器（类似于 MDX）的语法，可能不会在您的 Markdown 查看器中正确呈现。
-->
# CTRL
<div class="flex flex-wrap space-x-1"> <a href="https://huggingface.co/models?filter=ctrl"> <img alt="模型" src="https://img.shields.io/badge/所有模型页面-ctrl-blueviolet"> </a> <a href="https://huggingface.co/spaces/docs-demos/tiny-ctrl"> <img alt="空间" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"> </a> </div>

## 概览

CTRL 模型由 Nitish Shirish Keskar *，Bryan McCann*，Lav R. Varshney，Caiming Xiong 和 Richard Socher 在 [CTRL：用于可控生成的条件 Transformer 语言模型](https://arxiv.org/abs/1909.05858) 中提出。

它是一个因果（单向）的 Transformer，在非常大的语料库上进行了语言建模的预训练，语料库的大小约为 140GB，第一个令牌被保留为控制代码（例如链接、书籍、维基百科等）。

论文摘要如下所示：

*大规模语言模型显示出有希望的文本生成能力，但用户不能轻松地控制生成文本的特定方面。我们发布了 CTRL，一个拥有 16.3 亿参数的条件 Transformer 语言模型，训练时使用控制代码作为条件，控制模型的风格、内容和特定任务行为。控制代码是从自然与原始文本共现的结构中派生出来的，保留了无监督学习的优势，同时在文本生成中提供了更明确的控制。这些代码还允许 CTRL 预测给定序列的最可能部分训练数据。这为通过基于模型的源属性分析大量数据提供了一种潜在的方法。* 提示：
- CTRL 使用控制代码生成文本：它要求以某些单词、句子或链接开始生成连贯的文本。有关更多信息，请参阅 [原始实现](https://github.com/salesforce/ctrl)。  more information.
- CTRL 是一个具有绝对位置嵌入的模型，因此通常建议在右侧而不是  左侧填充输入。- CTRL 是使用因果语言建模（CLM）目标训练的，因此在预测序列中的下一个  令牌时非常强大。利用这个特性，CTRL 可以生成语法连贯的文本，如在  *run_generation.py* 示例脚本中所观察到的那样。- PyTorch 模型可以接受 `past_key_values` 作为输入，这是先前计算的  键/值注意力对。TensorFlow 模型接受 `past` 作为输入。使用 `past_key_values` 值  可以防止模型在文本生成的上下文中重新计算预先计算的值。

有关使用此参数的  更多信息，请参阅 [`forward`](model_doc/ctrl#transformers.CTRLModel.forward) 方法。

此模型由 [keskarnitishr](https://huggingface.co/keskarnitishr) 贡献。原始代码可以在 [这里](https://github.com/salesforce/ctrl) 找到。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)- [因果语言建模任务指南](../tasks/language_modeling)

## CTRLConfig

[[autodoc]] CTRLConfig

## CTRLTokenizer

[[autodoc]] CTRLTokenizer
    - save_vocabulary

## CTRLModel

[[autodoc]] CTRLModel
    - forward

## CTRLLMHeadModel

[[autodoc]] CTRLLMHeadModel
    - forward

## CTRLForSequenceClassification

[[autodoc]] CTRLForSequenceClassification
    - forward

## TFCTRLModel

[[autodoc]] TFCTRLModel
    - call

## TFCTRLLMHeadModel

[[autodoc]] TFCTRLLMHeadModel
    - call

## TFCTRLForSequenceClassification

[[autodoc]] TFCTRLForSequenceClassification
    - call
