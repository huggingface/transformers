<!-- 版权所有2022年HuggingFace团队保留所有权利。
根据Apache许可证第2.0版（“许可证”），除非符合许可证的要求，否则您不得使用此文件。您可以在许可证网站上获取许可证的副本。
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按照“原样”基础分发的，不附带任何形式的明示或暗示保证。请参阅许可证以了解特定语言下的权限和限制。⚠️请注意，此文件是Markdown格式，但包含我们文档构建器（类似于MDX）的特定语法，可能无法在您的Markdown查看器中正确显示。


-->

## 概览

REALM模型是由Kelvin Guu、Kenton Lee、Zora Tung、Panupong Pasupat和Ming-Wei Chang在[REALM：检索增强的语言模型预训练](https://arxiv.org/abs/2002.08909)中提出的。

它是一种检索增强的语言模型，首先从文本知识语料库中检索文档，然后利用检索到的文档来处理问题回答任务。

该论文的摘要如下：

*语言模型预训练已被证明可以捕捉大量的世界知识，这对于问答等NLP任务至关重要。然而，这些知识以神经网络的参数形式隐式存储，因此需要越来越大的网络来涵盖更多的事实。为了以更模块化和可解释的方式捕获知识，我们在语言模型预训练中增加了潜在的知识检索器，使模型能够从大规模语料库（如维基百科）中检索和关注文档，该语料库在预训练、微调和推理过程中使用。我们首次展示了如何以无监督的方式预训练这样的知识检索器，使用掩码语言建模作为学习信号，并通过考虑数百万个文档进行检索步骤的反向传播。我们通过在三个流行的开放领域问答基准测试中进行细调，对Retrieval-Augmented Language Model预训练（REALM）的有效性进行了验证。我们与所有先前方法相比，无论是显式还是隐式的知识存储，都取得了显著的提升（4-16%的绝对准确率），同时还提供了解释性和模块化等质量上的优势。*

此模型由[qqaatw](https://huggingface.co/qqaatw)贡献。原始代码可以在[这里](https://github.com/google-research/language/tree/master/language/realm)找到。
## RealmConfig

[[autodoc]] RealmConfig

## RealmTokenizer

[[autodoc]] RealmTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary
    - batch_encode_candidates

## RealmTokenizerFast

[[autodoc]] RealmTokenizerFast
    - batch_encode_candidates

## RealmRetriever

[[autodoc]] RealmRetriever

## RealmEmbedder

[[autodoc]] RealmEmbedder
    - forward

## RealmScorer

[[autodoc]] RealmScorer
    - forward

## RealmKnowledgeAugEncoder

[[autodoc]] RealmKnowledgeAugEncoder
    - forward

## RealmReader

[[autodoc]] RealmReader
    - forward

## RealmForOpenQA

[[autodoc]] RealmForOpenQA
    - block_embedding_to
    - forward