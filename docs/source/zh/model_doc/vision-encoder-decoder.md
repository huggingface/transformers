<!--版权所有 2021 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按照“按原样”基础分发的，不附带任何形式的保证或条件。请参阅许可证获取特定语言下的权限和限制。
⚠️请注意，此文件采用 Markdown 格式，但包含了我们的文档生成器的特定语法（类似于 MDX），因此在您的 Markdown 查看器中可能无法正确地渲染出来。
-->

# 视觉编码解码模型

## 概述

[`VisionEncoderDecoderModel`] 可用于使用任何预训练的基于 Transformer 的视觉模型作为编码器（*例如* [ViT](vit)，[BEiT](beit)，[DeiT](deit)，[Swin](swin)）和任何预训练的语言模型作为解码器（*例如* [RoBERTa](roberta)，[GPT2](gpt2)，[BERT](bert)，[DistilBERT](distilbert)）。在初始化图像到文本模型时，使用预训练的检查点来初始化模型已被证明是有效的（例如，参见 [Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang，Zhoujun Li, Furu Wei.的 TrOCR：基于 Transformer 的光学字符识别与预训练模型](https://arxiv.org/abs/2109.10282)）。

一旦这样的 [`VisionEncoderDecoderModel`] 已经被训练/微调，它可以像任何其他模型一样保存/加载（详见下面的示例了解更多信息）。example) [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282) by Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang,
Zhoujun Li, Furu Wei.

这样的 [`VisionEncoderDecoderModel`] 的一个示例应用是图像字幕，其中编码器用于编码图像，然后自回归语言模型生成字幕。另一个示例是光学字符识别。参见 [TrOCR](trocr)，它是 [`VisionEncoderDecoderModel`] 的一个实例。

## 从模型配置随机初始化 

`VisionEncoderDecoderModel`
[`VisionEncoderDecoderModel`] 可以从编码器和解码器的配置随机初始化。以下示例展示了如何使用默认的 [`ViTModel`] 配置来实现这一点，作为编码器和默认的 [`BertForCausalLM`] 配置作为解码器。
```python
>>> from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

>>> config_encoder = ViTConfig()
>>> config_decoder = BertConfig()

>>> config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = VisionEncoderDecoderModel(config=config)
```

## 从预训练的编码器和预训练的解码器 `VisionEncoderDecoderModel` 初始化 


[`VisionEncoderDecoderModel`] 可以从预训练的编码器检查点和预训练的解码器检查点初始化。请注意，任何预训练的基于 Transformer 的视觉模型（例如 [Swin](swin)）都可以作为编码器，并且任何预训练的自编码模型（例如 BERT）和预训练的因果语言模型（例如 GPT2），以及序列到序列模型（例如 BART 的解码器）的预训练解码器部分都可以用作解码器。根据您选择的解码器架构，交叉注意力层可能会被随机初始化。从预训练的编码器和解码器检查点初始化 [`VisionEncoderDecoderModel`] 需要在下游任务中对模型进行微调，如 [“启动编码器-解码器”，博客文章中所示](https://huggingface.co/blog/warm-starting-encoder-decoder)。

为此，`VisionEncoderDecoderModel` 类提供了一个 [`VisionEncoderDecoderModel.from_encoder_decoder_pretrained`] 方法。

```python
>>> from transformers import VisionEncoderDecoderModel

>>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "microsoft/swin-base-patch4-window7-224-in22k", "bert-base-uncased"
... )
```

## 加载现有的 `VisionEncoderDecoderModel` 检查点并进行推理

要加载 `VisionEncoderDecoderModel` 类的微调检查点，[`VisionEncoderDecoderModel`] 提供了与 Transformers 中的任何其他模型架构相同的 `from_pretrained(...)` 方法。

要进行推理，可以使用 [`generate`] 方法，该方法支持多种解码方式，如贪婪解码、束搜索和多项式采样。

```python
>>> import requests
>>> from PIL import Image

>>> from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

>>> # load a fine-tuned image captioning model and corresponding tokenizer and image processor
>>> model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
>>> tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
>>> image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

>>> # let's perform inference on an image
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values

>>> # autoregressively generate caption (uses greedy decoding by default)
>>> generated_ids = model.generate(pixel_values)
>>> generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
a cat laying on a blanket next to a cat laying on a bed
```

## 将 PyTorch 检查点加载到  `TFVisionEncoderDecoderModel`


[`TFVisionEncoderDecoderModel.from_pretrained`] 当前不支持使用 PyTorch 检查点初始化模型。将 `from_pt=True` 传递给该方法会引发异常。如果只有特定视觉编码器-解码器模型的 PyTorch 检查点，可以使用以下解决方法：

```python
>>> from transformers import VisionEncoderDecoderModel, TFVisionEncoderDecoderModel

>>> _model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

>>> _model.encoder.save_pretrained("./encoder")
>>> _model.decoder.save_pretrained("./decoder")

>>> model = TFVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "./encoder", "./decoder", encoder_from_pt=True, decoder_from_pt=True
... )
>>> # This is only for copying some specific attributes of this particular model.
>>> model.config = _model.config
```

## 训练

创建模型后，可以像 BART、T5 或任何其他编码器-解码器模型一样，对(image, text)对的数据集进行微调。如您所见，模型只需要 2 个输入来计算损失：`pixel_values`（即图像）和 `labels`（即编码后的目标序列的 `input_ids`）

```python
>>> from transformers import ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel
>>> from datasets import load_dataset

>>> image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
>>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
>>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "google/vit-base-patch16-224-in21k", "bert-base-uncased"
... )

>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values

>>> labels = tokenizer(
...     "an image of two cats chilling on a couch",
...     return_tensors="pt",
... ).input_ids

>>> # the forward function automatically creates the correct decoder_input_ids
>>> loss = model(pixel_values=pixel_values, labels=labels).loss
```

此模型由 [nielsr](https://github.com/nielsrogge) 贡献。

该模型的 TensorFlow 和 Flax 版本由 [ydshieh](https://github.com/ydshieh) 贡献。were contributed by [ydshieh](https://github.com/ydshieh).

## VisionEncoderDecoderConfig

[[autodoc]] VisionEncoderDecoderConfig

## VisionEncoderDecoderModel

[[autodoc]] VisionEncoderDecoderModel
    - forward
    - from_encoder_decoder_pretrained

## TFVisionEncoderDecoderModel

[[autodoc]] TFVisionEncoderDecoderModel
    - call
    - from_encoder_decoder_pretrained

## FlaxVisionEncoderDecoderModel

[[autodoc]] FlaxVisionEncoderDecoderModel
    - __call__
    - from_encoder_decoder_pretrained
