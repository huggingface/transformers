<!--版权所有2021年HuggingFace团队保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非符合许可证的规定，否则您不得使用本文件。您可以在下面获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，按“原样”分发的软件在许可证下分发，不附带任何形式的保证或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们 doc-builder（类似于 MDX）的特定语法，可能无法
在您的 Markdown 查看器中正确渲染。
-->
# XLS-R

## 概述

XLS-R 模型是由 Arun Babu、Changhan Wang、Andros Tjandra、Kushal Lakhotia、Qiantong Xu、NamanGoyal、Kritika Singh、Patrick von Platen、Yatharth Saraf、Juan Pino、Alexei Baevski、Alexis Conneau 和 Michael Auli 在 [XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale](https://arxiv.org/abs/2111.09296) 中提出的。

论文摘要如下：

*本文介绍了基于 wav2vec 2.0 的大规模跨语言语音表示学习模型 XLS-R。我们在 128 个语言中使用高达 20 亿参数的模型进行训练，使用近五十万小时的公开语音音频数据，比已知最大的先前工作多了一个数量级的数据。我们的评估涵盖了广泛的任务、领域、数据范围和语言，包括高资源和低资源情况。在 CoVoST-2 语音翻译基准测试中，我们相对于 21 个翻译方向的最佳已知先前工作平均提高了 7.4 个 BLEU。对于语音识别，XLS-R 相对于 BABEL、MLS、CommonVoice 以及 VoxPopuli 等已知最佳先前工作平均降低了 14-34%的错误率。XLS-R 还在语言识别方面创造了新的最佳成绩。此外，我们还展示了在翻译英语语音到其他语言时，跨语言预训练在足够大的模型尺寸下可以优于仅英语预训练的情况，这种情况有利于单语预训练。我们希望 XLS-R 可以帮助改进更多世界语言的语音处理任务。*

提示：

- XLS-R 是一个接受与语音信号的原始波形相对应的浮点数组的语音模型。- XLS-R 模型是使用连接主义时间分类（CTC）训练的，因此必须使用 [`Wav2Vec2CTCTokenizer`] 对模型输出进行解码。  [`Wav2Vec2CTCTokenizer`].

相关检查点可以在 https://huggingface.co/models?other = xls_r 找到。

XLS-R 的架构基于 Wav2Vec2 模型，因此可以参考 [Wav2Vec2 的文档页面](wav2vec2)。

原始代码可在 [此处](https://github.com/pytorch/fairseq/tree/master/fairseq/models/wav2vec) 找到。