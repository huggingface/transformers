<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
# Wav2Vec2Phoneme

## 概述

Wav2Vec2Phoneme 模型是由 Qiantong Xu，Alexei Baevski 和 Michael Auli 于 2021 年提出的 [简单高效的零样本跨语言音素识别（Xu 等，2021）](https://arxiv.org/abs/2109.11680)。

以下是摘自论文的摘要:

*最近在自训练、自监督预训练和无监督学习方面取得的进展使得在没有任何标记数据的情况下能够实现良好的语音识别系统。然而，在许多情况下，存在与相关语言相关的标记数据，但这些方法并未利用该数据。本文通过微调多语言预训练的 wav2vec 2.0 模型来转录未知语言。这是通过使用发音特征将训练语言的音素映射到目标语言来实现的。实验证明，这种简单的方法明显优于之前引入了任务特定体系结构并仅使用了预训练模型的部分内容的工作。* 



提示：

- Wav2Vec2Phoneme 与 Wav2Vec2 完全使用相同的架构- Wav2Vec2Phoneme 是一个语音模型，它接受与语音信号的原始波形相对应的浮点数组。- Wav2Vec2Phoneme 模型使用连续时间分类（CTC）进行训练，因此模型输出必须使用 [`Wav2Vec2PhonemeCTCTokenizer`] 进行解码。
- Wav2Vec2Phoneme 可以同时在多种语言上进行微调，并在单次前向传递中对未知语言进行解码，得到一系列的音素。- 默认情况下，该模型输出一系列的音素。

为了将音素转换为一系列的词语，应使用字典和语言模型。 

相关的检查点可以在 https://huggingface.co/models?other = phoneme-recognition 中找到。

此模型由 [patrickvonplaten](https://huggingface.co/patrickvonplaten) 贡献

原始代码可以在 [此处](https://github.com/pytorch/fairseq/tree/master/fairseq/models/wav2vec) 找到。

Wav2Vec2Phoneme 的架构基于 Wav2Vec2 模型，因此可以参考 [`Wav2Vec2`] 的文档页面，除了分词器 (Tokenizer)（tokenizer）。

## Wav2Vec2PhonemeCTCTokenizer

[[autodoc]] Wav2Vec2PhonemeCTCTokenizer
	- __call__
	- batch_decode
	- decode
	- phonemize