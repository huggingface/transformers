<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证 2.0 版（“许可证”）授权；您除非符合许可证，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按照“按原样”基础分发的，不附带任何明示或暗示的担保或条件。请参阅许可证特定语言下的权限和限制。
⚠️请注意，此文件是 Markdown 格式，但包含特定于我们的文档生成器（类似于 MDX）的语法，可能不会在您的 Markdown 查看器中正确呈现。
-->
# LiLT

## 概览

LiLT 模型是由 Jiapeng Wang、Lianwen Jin 和 Kai Ding 在 [LiLT：一种简单而有效的语言无关布局 Transformer 用于结构化文档理解](https://arxiv.org/abs/2202.13669) 中提出的。LiLT 允许将任何预训练的 RoBERTa 文本编码器与轻量级布局 Transformer 相结合，以实现类似于 [LayoutLM](layoutlm) 的多语言文档理解语言。

论文的摘要如下所示：

*结构化文档理解近来引起了广泛关注并取得了重要进展，这要归功于其在智能文档处理中的关键作用。然而，大多数现有的相关模型只能处理特定语言（通常是英语）的文档数据，这极大地限制了应用范围。为解决这个问题，我们提出了一种简单而有效的面向语言无关的布局 Transformer（LiLT）用于结构化文档理解。LiLT 可以在单一语言的结构化文档上进行预训练，然后使用相应的现成的单语/多语预训练文本模型进行直接微调以适用于其他语言。在八种语言上的实验结果表明，LiLT 在多样化的广泛用途下游基准测试中可以取得竞争性甚至更优的性能，从而使得可以从文档布局结构的预训练中获益并且不受语言限制。*

提示：

- 要将语言无关的布局 Transformer 与新的来自 [hub](https://huggingface.co/models?search=roberta) 的 RoBERTa 检查点相结合，请参考 [此指南](https://github.com/jpWang/LiLT#or-generate-your-own-checkpoint-optional)。运行脚本将会在本地存储 `config.json` 和 `pytorch_model.bin` 文件。完成后，您可以执行以下操作（假设您已使用 HuggingFace 帐户登录）：
```
from transformers import LiltModel

model = LiltModel.from_pretrained("path_to_your_files")
model.push_to_hub("name_of_repo_on_the_hub")
```

- 在为模型准备数据时，请确保使用与您与布局 Transformer 相结合的 RoBERTa 检查点相对应的令牌词汇表。- 由于 [lilt-roberta-en-base](https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base) 使用与 [LayoutLMv3](layoutlmv3) 相同的词汇表，因此可以使用 [`LayoutLMv3TokenizerFast`] 为模型准备数据。

同样对于 [lilt-roberta-en-base](https://huggingface.co/SCUT-DLVCLab/lilt-infoxlm-base) 也是如此：可以使用 [`LayoutXLMTokenizerFast`] 进行准备。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/lilt_architecture.jpg"alt="drawing" width="600"/>

<small> LiLT 架构。摘自 <a href="https://arxiv.org/abs/2202.13669"> 原始论文 </a>。</small>
此模型由 [nielsr](https://huggingface.co/nielsr) 贡献。可以在 [此处](https://github.com/jpwang/lilt) 找到原始代码。

## 资源

以下是官方 Hugging Face 和社区（由🌎表示）资源列表，可帮助您开始使用 LiLT。
- LiLT 的演示笔记本可以在 [此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LiLT) 找到。

**文档资源**
- [文本分类任务指南](../tasks/sequence_classification)
- [令牌分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)

如果您有兴趣提交资源以包含在此处，请随时打开拉取请求，我们将对其进行审查！该资源应该展示出新的东西，而不是重复现有的资源。

## LiltConfig

[[autodoc]] LiltConfig

## LiltModel

[[autodoc]] LiltModel
    - forward

## LiltForSequenceClassification

[[autodoc]] LiltForSequenceClassification
    - forward

## LiltForTokenClassification

[[autodoc]] LiltForTokenClassification
    - forward

## LiltForQuestionAnswering

[[autodoc]] LiltForQuestionAnswering
    - forward
