<!--版权 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非遵守许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按“按原样”基础分发的，不附带任何形式的担保或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️ 请注意，此文件是使用 Markdown 编写的，但包含我们 doc-builder（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。
-->
# SwitchTransformers

## 概述

SwitchTransformers 模型是由 William Fedus、Barret Zoph 和 Noam Shazeer 在 [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) 中提出的。

Switch Transformer 模型使用稀疏的 T5 编码器-解码器架构，其中 MLP 被 Mixture of Experts（MoE）替代。路由机制（此处为 top 1）将每个令牌与一个专家相关联，其中每个专家都是一个密集 MLP。尽管 switch transformers 的权重比其等效的密集模型多得多，但稀疏性可以实现更好的扩展和更好的规模微调性能。在前向传递过程中，只使用权重的一小部分。路由机制允许模型实时选择相关权重，从而增加模型容量而不增加操作数。

论文中的摘要如下：

*在深度学习中，模型通常对所有输入重复使用相同的参数。Mixture of Experts（MoE）打破了这一点，它为每个输入示例选择不同的参数。结果是一个稀疏激活的模型——具有大量参数——但计算成本恒定。然而，尽管 MoE 取得了一些显著的成功，但它的广泛应用受到了复杂性、通信成本和训练不稳定性的阻碍——我们通过 Switch Transformer 解决了这些问题。我们简化了 MoE 路由算法，并设计了直观的改进模型，减少了通信和计算成本。我们提出的训练技术有助于解决不稳定性问题，并且我们展示了大规模稀疏模型可以使用更低精度（bfloat16）格式训练。我们基于 T5-Base 和 T5-Large 设计的模型，在相同的计算资源下，可以实现高达 7 倍的预训练速度提升。这些改进在多语言环境中也适用，我们在所有 101 种语言上均对 mT5-Base 版本进行了增益测量。最后，我们通过对“Colossal Clean Crawled Corpus”进行预训练，将语言模型的当前规模推进到了万亿参数模型，并实现了对 T5-XXL 模型的 4 倍加速。*

提示：

- SwitchTransformers 使用 [`T5Tokenizer`]，可以直接从每个模型的存储库加载。
- 发布的权重是在英语 [Masked Language Modeling](https://moon-ci-docs.huggingface.co/docs/transformers/pr_19323/en/glossary#general-terms) 任务上进行预训练的，需要进行微调。

该模型由 [Younes Belkada](https://huggingface.co/ybelkada) 和 [Arthur Zucker](https://huggingface.co/ArtZucker) 贡献。原始代码可以在 [此处](https://github.com/google/flaxformer/tree/main/flaxformer/architectures/moe) 找到。

## 资源

- [翻译任务指南](../tasks/translation)
- [摘要任务指南](../tasks/summarization)

## SwitchTransformersConfig

[[autodoc]] SwitchTransformersConfig

## SwitchTransformersTop1Router

[[autodoc]] SwitchTransformersTop1Router
    - _compute_router_probabilities
    - forward

## SwitchTransformersSparseMLP

[[autodoc]] SwitchTransformersSparseMLP
    - forward

## SwitchTransformersModel

[[autodoc]] SwitchTransformersModel
    - forward

## SwitchTransformersForConditionalGeneration

[[autodoc]] SwitchTransformersForConditionalGeneration
    - forward

## SwitchTransformersEncoderModel

[[autodoc]] SwitchTransformersEncoderModel
    - forward
