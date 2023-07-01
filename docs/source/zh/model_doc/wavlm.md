<!--版权所有 2021 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的要求，否则您不得使用此文件。您可以在许可证网址中获得许可证的副本。
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，按“原样”基础分发的软件不附带任何形式的担保或条件，无论是明示的还是暗示的。请参阅许可证了解特定语言下权限和限制的详细信息。请注意，该文件是 Markdown 格式的，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中
正确显示。-->

# WavLM

## 概述

WavLM 模型是由 Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Furu Wei 在 [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) 中提出的。以下是论文中的摘要: Michael Zeng, Furu Wei.

*自我监督学习（Self-supervised learning，SSL）在语音识别方面取得了巨大成功，但在其他语音处理任务方面的尝试有限。由于语音信号包含说话人身份、语用学特征、言语内容等多方面的信息，学习适用于所有语音任务的通用表示是具有挑战性的。在本文中，我们提出了一种新的预训练模型 WavLM，用于解决全栈下游语音任务。WavLM 基于 HuBERT 框架构建，注重言语内容建模和说话人身份保持两方面。我们首先在 Transformer 结构中引入门控相对位置偏置，以提高其在识别任务上的能力。为了更好地区分说话人，我们提出了一种语音混合训练策略，即无监督创建额外重叠的语音片段，并在模型训练过程中加以融合。最后，我们将训练数据集从 60,000 小时扩展到 94,000 小时。WavLM Large 在 SUPERB 基准测试中实现了最先进的性能，并为各种语音处理任务在其代表性基准测试中带来了显著的改进。*

提示：

- WavLM 是一个接受与语音信号的原始波形相对应的浮点数组的语音模型。请使用 [`Wav2Vec2Processor`] 进行特征提取。  

- WavLM 模型可以使用连接主义时间分类（CTC）进行微调，所以模型输出必须使用 [`Wav2Vec2CTCTokenizer`] 进行解码。- WavLM 在说话人验证、说话人识别和说话人分割等任务上表现特别出色。

相关检查点可在 https://huggingface.co/models?other = wavlm 中找到。

此模型由 [patrickvonplaten](https://huggingface.co/patrickvonplaten) 提供。作者的代码可在 [此处](https://github.com/microsoft/unilm/tree/master/wavlm) 找到。

## 文档资源

- [音频分类任务指南](../tasks/audio_classification)

- [自动语音识别任务指南](../tasks/asr)
## WavLMConfig

[[autodoc]] WavLMConfig

## WavLMModel

[[autodoc]] WavLMModel
    - forward

## WavLMForCTC

[[autodoc]] WavLMForCTC
    - forward

## WavLMForSequenceClassification

[[autodoc]] WavLMForSequenceClassification
    - forward

## WavLMForAudioFrameClassification

[[autodoc]] WavLMForAudioFrameClassification
    - forward

## WavLMForXVector

[[autodoc]] WavLMForXVector
    - forward
