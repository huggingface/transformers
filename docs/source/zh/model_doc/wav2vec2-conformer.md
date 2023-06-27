<!--版权 2022 年由 HuggingFace 团队保留。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；您除非符合许可证的要求，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“按原样” BASIS，无论是明示还是暗示的，都没有任何保证或条件。请参阅许可证具体语言下的权限和限制。
⚠️ 请注意，此文件是 Markdown 格式的，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确显示。
-->

# Wav2Vec2-Conformer

## Overview## 概述（Overview）

Wav2Vec2-Conformer 是由 Changhan Wang、Yun Tang、Xutai Ma、Anne Wu、Sravya Popuri、Dmytro Okhonko、Juan Pino 在更新版本的 [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://arxiv.org/abs/2010.05171) 中添加的。

该模型的官方结果可在论文的第 3 表和第 4 表中找到。
Wav2Vec2-Conformer 的权重由 Meta AI 团队在 [Fairseq 库](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/README.md#pre-trained-models) 中发布。

提示：

- Wav2Vec2-Conformer 遵循与 Wav2Vec2 相同的架构，但将 *Attention*-block 替换为 *Conformer*-block，- Wav2Vec2-Conformer follows the same architecture as Wav2Vec2, but replaces the *Attention*-block with a *Conformer*-block 即在 [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) 中引入。 
- 对于相同数量的层，Wav2Vec2-Conformer 需要更多的参数，但也能得到更低的词错误率。
- For the same number of layers, Wav2Vec2-Conformer requires more parameters than Wav2Vec2, but also yields 改进的词错误率。an improved word error rate.
- Wav2Vec2-Conformer 使用与 Wav2Vec2 相同的分词器 (Tokenizer)和特征提取器。

- Wav2Vec2-Conformer 可以使用无相对位置嵌入、类似 Transformer-XL 的位置嵌入或旋转位置嵌入，只需设置正确的 `config.position_embeddings_type`。

该模型由 [patrickvonplaten](https://huggingface.co/patrickvonplaten) 贡献。

原始代码可以在 [此处](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec) 找到。



## 文档资源（Documentation resources）

- [音频分类任务指南](../tasks/audio_classification)

- [自动语音识别任务指南](../tasks/asr)- [Automatic speech recognition task guide](../tasks/asr)

## Wav2Vec2ConformerConfig

[[autodoc]] Wav2Vec2ConformerConfig

## Wav2Vec2Conformer specific outputs

[[autodoc]] models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForPreTrainingOutput

## Wav2Vec2ConformerModel

[[autodoc]] Wav2Vec2ConformerModel
    - forward

## Wav2Vec2ConformerForCTC

[[autodoc]] Wav2Vec2ConformerForCTC
    - forward

## Wav2Vec2ConformerForSequenceClassification

[[autodoc]] Wav2Vec2ConformerForSequenceClassification
    - forward

## Wav2Vec2ConformerForAudioFrameClassification

[[autodoc]] Wav2Vec2ConformerForAudioFrameClassification
    - forward

## Wav2Vec2ConformerForXVector

[[autodoc]] Wav2Vec2ConformerForXVector
    - forward

## Wav2Vec2ConformerForPreTraining

[[autodoc]] Wav2Vec2ConformerForPreTraining
    - forward
