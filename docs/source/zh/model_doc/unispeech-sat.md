<!--版权所有 2021 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合许可证，否则您不得使用此文件。您可以在
http://www.apache.org/licenses/LICENSE-2.0
适用法律要求或以书面形式同意的情况下，按原样分发的软件基础上，无论是明示还是暗示，都不提供任何担保或条件。请参阅许可证特定语言的权限和限制。
⚠️请注意，此文件是 Markdown 格式的，但包含我们的文档生成器（类似于 MDX）的特定语法，可能在您的 Markdown 查看器中无法正确呈现。
-->
# UniSpeech-SAT

## 概述

UniSpeech-SAT 模型是由 Sanyuan Chen，Yu Wu，Chengyi Wang，Zhengyang Chen，Zhuo Chen，Shujie Liu，Jian Wu，Yao Qian，Furu Wei，Jinyu Li，Xiangzhan Yu 在 [UniSpeech-SAT: Universal Speech Representation Learning with Speaker AwarePre-Training](https://arxiv.org/abs/2110.05752) 提出的。

该论文的摘要如下：
*自我监督学习（SSL）是语音处理的长期目标，因为它利用大规模的无标签

*近年来，在语音识别中应用自我监督学习取得了巨大成功而在应用 SSL 进行建模说话者特征方面进行了有限的探索。本文旨在改进现有的 SSL 框架以进行说话者表示学习。我们提出了两种方法来增强无监督说话者信息提取。首先，我们将多任务学习应用于当前的 SSL 框架，我们将话语级对比损失与 SSL 目标函数相结合。其次，为了更好地区分说话者，我们提出了一种话语混合策略进行数据增强，在训练过程中无监督地创建额外的重叠话语并进行合并。我们将提出的方法集成到 HuBERT 框架中。在 SUPERB 基准测试中的实验证明，所提出的系统实现了在通用表示学习方面的最新性能，特别是针对说话者识别。进行了消融研究 验证了每种提出方法的功效。最后，我们将训练数据扩大到 94,000 小时的公共音频数据，并在所有 SUPERB 任务中取得了进一步的性能提升。*

提示：

- UniSpeechSat 是一个接受与语音信号的原始波形对应的浮点数组的语音模型。  请使用 [`Wav2Vec2Processor`] 进行特征提取。

- UniSpeechSat 模型可以使用连续时间分类（CTC）进行微调，因此必须对模型输出进行  解码，使用 [`Wav2Vec2CTCTokenizer`]。

- UniSpeechSat 在说话者验证、说话者识别和说话者分离任务上表现特别好。

此模型由 [patrickvonplaten](https://huggingface.co/patrickvonplaten) 贡献。作者的代码可以在此处找到（https://github.com/microsoft/UniSpeech/tree/main/UniSpeech-SAT）。


## 文档资源

- [音频分类任务指南](../tasks/audio_classification)
- [自动语音识别任务指南](../tasks/asr)

## UniSpeechSatConfig

[[autodoc]] UniSpeechSatConfig

## UniSpeechSat specific outputs

[[autodoc]] models.unispeech_sat.modeling_unispeech_sat.UniSpeechSatForPreTrainingOutput

## UniSpeechSatModel

[[autodoc]] UniSpeechSatModel
    - forward

## UniSpeechSatForCTC

[[autodoc]] UniSpeechSatForCTC
    - forward

## UniSpeechSatForSequenceClassification

[[autodoc]] UniSpeechSatForSequenceClassification
    - forward

## UniSpeechSatForAudioFrameClassification

[[autodoc]] UniSpeechSatForAudioFrameClassification
    - forward

## UniSpeechSatForXVector

[[autodoc]] UniSpeechSatForXVector
    - forward

## UniSpeechSatForPreTraining

[[autodoc]] UniSpeechSatForPreTraining
    - forward
