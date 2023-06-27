<!--版权所有2021年HuggingFace团队保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的保证或条件。请参阅许可证以了解特定语言下的权限和限制。⚠️请注意，此文件使用 Markdown 编写，但包含特定的语法以适应我们的文档构建程序（类似于 MDX），因此在 Markdown 阅读器中可能无法正确显示。
-->

# SEW-D

## 概述

SEW-D（Squeezed and Efficient Wav2Vec with Disentangled attention）是由 Felix Wu，Kwangyoun Kim，Jing Pan，Kyu Han，Kilian Q. Weinberger 和 Yoav Artzi 在 [Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/2109.06870) 中提出的。

论文摘要如下：

*本论文研究了预训练语音识别（ASR）模型的性能和效率之间的权衡。我们着重研究了 wav2vec 2.0，并制定了几种影响模型性能和效率的架构设计。通过综合我们所有的观察结果，我们引入了 SEW（Squeezed and Efficient Wav2vec），这是一个在性能和效率两个维度上都有显著改进的预训练模型架构，适用于多种训练设置。例如，在 LibriSpeech 的 100h-960h 半监督设置下，与 wav2vec 2.0 相比，SEW 的推理速度提高了 1.9 倍，相对于词错误率减少了 13.5%。在相似的推理时间下，SEW 在不同模型尺寸上将词错误率降低了 25-50%。*

提示：

- SEW-D 是一个接受与语音信号的原始波形对应的浮点数组的语音模型。- SEWDForCTC 使用连接主义时间分类（CTC）进行微调，因此必须使用 [`Wav2Vec2CTCTokenizer`] 对模型输出进行解码。  using [`Wav2Vec2CTCTokenizer`].

该模型由 [anton-l](https://huggingface.co/anton-l) 贡献。

## 文档资源

- [音频分类任务指南](../tasks/audio_classification)

- [自动语音识别任务指南](../tasks/asr)

## SEWDConfig

[[autodoc]] SEWDConfig

## SEWDModel

[[autodoc]] SEWDModel
    - forward

## SEWDForCTC

[[autodoc]] SEWDForCTC
    - forward

## SEWDForSequenceClassification

[[autodoc]] SEWDForSequenceClassification
    - forward