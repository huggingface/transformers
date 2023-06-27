<!--版权所有 2020 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非符合许可证，否则您不得使用此文件。您可以在
http://www.apache.org/licenses/LICENSE-2.0
适用的法律要求或书面同意的情况下，根据许可证分发的软件以“按原样”基础分发，不附带任何形式的担保或条件。请参阅许可证以获取特定语言的权限和限制。
⚠️ 请注意，该文件是 Markdown 格式，但包含我们 doc-builder（类似 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。
-->
# DPR
<div class="flex flex-wrap space-x-1"> <a href="https://huggingface.co/models?filter=dpr"> <img alt="Models" src="https://img.shields.io/badge/All_model_pages-dpr-blueviolet"> </a> <a href="https://huggingface.co/spaces/docs-demos/dpr-question_encoder-bert-base-multilingual"> <img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"> </a> </div>

## 概述

Dense Passage Retrieval（DPR）是一套用于最先进的开放领域问答（Q&A）研究的工具和模型。它是由 Vladimir Karpukhin，Barlas O ğ uz，Sewon Min，Patrick Lewis，Ledell Wu，Sergey Edunov，Danqi Chen 和 Wen-tau Yih 在 [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) 中提出的。

论文中的摘要如下：
*开放领域问答依赖于高效的段落检索来选择候选上下文，传统 TF-IDF 或 BM25 等稀疏向量空间模型是事实上的方法。在这项工作中，我们展示了检索可以仅使用密集表示来实现，其中嵌入是通过一个简单的双编码器框架从一小部分问题和段落中学习的。在广泛的开放领域 QA 数据集上进行评估时，我们的密集检索器在排名前 20 的段落的检索准确性方面基本上优于强大的 Lucene-BM25 系统，提高了 9%-19%的绝对值，并且帮助我们的端到端 QA 系统在多个开放领域 QA 基准上建立了新的最先进技术。*
此模型由 [lhoestq](https://huggingface.co/lhoestq) 贡献。原始代码可以在 [这里](https://github.com/facebookresearch/DPR) 找到。

提示：

- DPR 由三个模型组成：
    * 问题编码器：将问题编码为向量    
    * 上下文编码器：将上下文编码为向量    
    * 阅读器：从检索到的上下文中提取问题的答案，并附带相关度得分（如果推断的范围实际上回答了问题，则得分较高）。
## DPRConfig
[[autodoc]] DPRConfig
## DPRContextEncoderTokenizer
[[autodoc]] DPRContextEncoderTokenizer
## DPRContextEncoderTokenizerFast
[[autodoc]] DPRContextEncoderTokenizerFast
## DPRQuestionEncoderTokenizer
[[autodoc]] DPRQuestionEncoderTokenizer
## DPRQuestionEncoderTokenizerFast
[[autodoc]] DPRQuestionEncoderTokenizerFast
## DPRReaderTokenizer
[[autodoc]] DPRReaderTokenizer
## DPRReaderTokenizerFast
[[autodoc]] DPRReaderTokenizerFast

## DPR 特定输出
[[autodoc]] models.dpr.modeling_dpr.DPRContextEncoderOutput
[[autodoc]] models.dpr.modeling_dpr.DPRQuestionEncoderOutput
[[autodoc]] models.dpr.modeling_dpr.DPRReaderOutput
## DPRContextEncoder

[[autodoc]] DPRContextEncoder
    - forward

## DPRQuestionEncoder

[[autodoc]] DPRQuestionEncoder
    - forward

## DPRReader

[[autodoc]] DPRReader
    - forward

## TFDPRContextEncoder

[[autodoc]] TFDPRContextEncoder
    - call

## TFDPRQuestionEncoder

[[autodoc]] TFDPRQuestionEncoder
    - call

## TFDPRReader

[[autodoc]] TFDPRReader
    - call
