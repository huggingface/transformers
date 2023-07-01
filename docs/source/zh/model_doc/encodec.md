<!--版权所有 2023 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下网址获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“原样”分发，不附带任何明示或暗示的保证或条件。请参阅许可证的具体语言以了解权限和限制。⚠️ 请注意，此文件为 Markdown 格式，但包含我们文档构建器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确呈现。-->
# EnCodec

## 概述 

EnCodec 神经编解码模型是由 Alexandre D é fossez、Jade Copet、Gabriel Synnaeve 和 Yossi Adi 在 [高保真神经音频压缩](https://arxiv.org/abs/2210.13438) 中提出的。

下面是来自论文的摘要：

*我们引入了一种最先进的实时高保真音频编解码器，利用神经网络。它由一个流式编码器-解码器架构组成，其中的量化潜空间是以端到端方式训练的。我们通过使用单个多尺度频谱对抗器来简化和加速训练，该对抗器能够有效减少伪影并产生高质量的样本。我们引入了一种新的损失平衡机制来稳定训练：现在损失的权重定义了它应该代表的整体梯度的比例，从而将这个超参数的选择与损失的典型尺度解耦。最后，我们研究了如何使用轻量级 Transformer 模型进一步压缩所得到的表示，压缩率高达 40%，同时保持快于实时。我们对所提出模型的关键设计选择进行了详细描述，包括：训练目标、架构变化以及各种感知损失函数的研究。我们进行了广泛的主观评估（MUSHRA 测试），并对各种带宽和音频领域进行了消融研究，包括语音、带噪混响语音和音乐。在所有评估的设置中，我们的方法优于基准方法，考虑到 24 kHz 单声道和 48 kHz 立体声音频。*

[Matthijs](https://huggingface.co/Matthijs)、[Patrick Von Platen](https://huggingface.co/patrickvonplaten) 和 [Arthur Zucker](https://huggingface.co/ArthurZ) 贡献了这个模型。


原始代码可在 [此处](https://github.com/facebookresearch/encodec) 找到。以下是使用此模型对音频进行编码和解码的快速示例：
```python 
>>> from datasets import load_dataset, Audio
>>> from transformers import EncodecModel, AutoProcessor
>>> librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> model = EncodecModel.from_pretrained("facebook/encodec_24khz")
>>> processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
>>> librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
>>> audio_sample = librispeech_dummy[-1]["audio"]["array"]
>>> inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

>>> encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
>>> audio_values = model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]
>>> # or the equivalent with a forward pass
>>> audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values
```


## EncodecConfig

[[autodoc]] EncodecConfig

## EncodecFeatureExtractor

[[autodoc]] EncodecFeatureExtractor
    - __call__

## EncodecModel

[[autodoc]] EncodecModel
    - decode
    - encode
    - forward