<!--版权所有 2021 年 NVIDIA Corporation 和 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下网址获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按原样的方式分发的，不附带任何明示或暗示的担保或条件。请参阅许可证获取特定语言下的权限和限制。特别提示：
⚠️请注意，此文件是 Markdown 格式的，但包含了我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。
-->
# MegatronBERT

## 概述

MegatronBERT 模型是由 Mohammad Shoeybi、Mostofa Patwary、Raul Puri、Patrick LeGresley、Jared Casper 和 Bryan Catanzaro 提出的 [《Megatron-LM: 使用模型并行训练数十亿参数的语言模型》](https://arxiv.org/abs/1909.08053)。

该论文的摘要如下：

*最新的语言模型研究表明，训练大型 Transformer 模型可以推动自然语言处理应用的技术进步。然而，非常大的模型由于内存限制往往很难训练。在本研究中，我们提出了一种训练非常大的 Transformer 模型的技术，并实现了一种简单高效的层内模型并行方法，可以使用数十亿个参数进行 Transformer 模型的训练。我们的方法不需要新的编译器或库更改，与流水线模型并行无关，可以通过在原生 PyTorch 中插入少量通信操作来完全实现。我们使用 512 个 GPU 收敛了基于 Transformer 的模型，参数量达到 83 亿。在整个应用程序上，我们实现了 15.1 PetaFLOPs 的性能，相对于单个 GPU 基准的 76%的扩展效率，而单个 GPU 基准维持了 39 TeraFLOPs 的性能，即其峰值 FLOPs 的 30%。为了证明大型语言模型可以进一步推动技术进步，我们训练了一个与 GPT-2 类似的 83 亿参数的 Transformer 语言模型以及一个与 BERT 类似的 39 亿参数的模型。我们展示了在 BERT 类模型中，对层归一化的精确放置对于实现随着模型规模增长而提高性能至关重要。在 WikiText103（困惑度从 15.8 降至 10.8）和 LAMBADA（准确率从 SOTA 的 63.2%提高到 66.5%）数据集上，我们使用 GPT-2 模型取得了 SOTA 结果。在 RACE 数据集上，我们的 BERT 模型取得了 SOTA 结果（准确率从 89.4%提高到 90.9%）。*

提示：

我们提供了预训练的 [BERT-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_bert_345m) 检查点，用于评估或微调后续任务。要访问这些检查点，首先 [注册](https://ngc.nvidia.com/signup) 并设置 NVIDIA GPU 云（NGC）注册表 CLI。有关下载模型的更多文档，请参阅 [NGC 文档](https://docs.nvidia.com/dgx/ngc-registry-cli-user-guide/index.html#topic_6_4_1)。

或者，您可以直接使用以下命令下载检查点：

BERT-345M-uncased:

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_uncased/zip
-O megatron_bert_345m_v0_1_uncased.zip
```

BERT-345M-cased:

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip -O
megatron_bert_345m_v0_1_cased.zip
```

一旦您从 NVIDIA GPU 云（NGC）获得了检查点，您需要将它们转换为 Hugging Face Transformers 和我们的 BERT 代码的可加载格式。以下命令使您可以进行转换。我们假设文件夹 `models/megatron_bert` 包含 `megatron_bert_345m_v0_1_{cased, uncased}.zip`，并且命令是在该文件夹内运行的。



```bash
python3 $PATH_TO_TRANSFORMERS/models/megatron_bert/convert_megatron_bert_checkpoint.py megatron_bert_345m_v0_1_uncased.zip
```

```bash
python3 $PATH_TO_TRANSFORMERS/models/megatron_bert/convert_megatron_bert_checkpoint.py megatron_bert_345m_v0_1_cased.zip
```

该模型由 [jdemouth](https://huggingface.co/jdemouth) 贡献。原始代码可以在 [此处](https://github.com/NVIDIA/Megatron-LM) 找到。该存储库包含了 Megatron 语言模型的多 GPU 和多节点实现。特别是，它包含了一种使用“张量并行”和“流水线并行”技术的混合模型并行方法。


## 文档资源
- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)- [因果语言建模任务指南](../tasks/language_modeling)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## MegatronBertConfig

[[autodoc]] MegatronBertConfig

## MegatronBertModel

[[autodoc]] MegatronBertModel
    - forward

## MegatronBertForMaskedLM

[[autodoc]] MegatronBertForMaskedLM
    - forward

## MegatronBertForCausalLM

[[autodoc]] MegatronBertForCausalLM
    - forward

## MegatronBertForNextSentencePrediction

[[autodoc]] MegatronBertForNextSentencePrediction
    - forward

## MegatronBertForPreTraining

[[autodoc]] MegatronBertForPreTraining
    - forward

## MegatronBertForSequenceClassification

[[autodoc]] MegatronBertForSequenceClassification
    - forward

## MegatronBertForMultipleChoice

[[autodoc]] MegatronBertForMultipleChoice
    - forward

## MegatronBertForTokenClassification

[[autodoc]] MegatronBertForTokenClassification
    - forward

## MegatronBertForQuestionAnswering

[[autodoc]] MegatronBertForQuestionAnswering
    - forward
