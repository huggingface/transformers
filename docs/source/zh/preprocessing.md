<!--版权所有 2023 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”），您只有在遵守许可证的情况下才能使用此文件许可证。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律或书面同意，根据许可证分发的软件均按照“按原样”基础分发，无论是以明示还是默示的方式。请参阅许可证以获取特定语言下的权限和限制。
⚠️ 注意，此文件是 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。
-->

# 预处理

[[在 Colab 中打开]]

在对数据集进行模型训练之前，需要将其预处理为预期的模型输入格式。无论您的数据是文本、图像还是音频，都需要将其转换和组装为张量批次。🤗 Transformers 提供了一组预处理类，以帮助准备数据以供模型使用。在本教程中，您将了解以下内容：

* 对于文本，请使用 [Tokenizer](./main_classes/tokenizer) 将文本转换为标记序列，创建标记的数值表示，并将其组装为张量。
* 对于语音和音频，请使用 [Feature extractor](./main_classes/feature_extractor) 从音频波形中提取连续特征并将其转换为张量。
* 对于图像输入，请使用 [ImageProcessor](./main_classes/image) 将图像转换为张量。
* 对于多模态输入，请使用 [Processor](./main_classes/processors) 组合标记器和特征提取器或图像处理器 (Image Processor)。
<Tip>

`AutoProcessor` 始终有效，并自动选择适用于您正在使用的模型的正确类别，无论您是使用令牌化器、图像处理器 (Image Processor)、特征提取器还是处理器。
</Tip>

开始之前，请安装🤗数据集，以便您可以加载一些数据集进行实验：
```bash
pip install datasets
```

## 自然语言处理
<Youtube id="Yffk5aydLzg"/>

预处理文本数据的主要工具是 [Tokenizer](main_classes/tokenizer)。

Tokenizer 根据一组规则将文本拆分为 *标记*。然后将这些标记转换为数字，然后转换为张量，成为模型的输入。令牌化器还会添加模型所需的任何其他输入。
<Tip>

如果您计划使用预训练模型，则使用相应的预训练令牌化器非常重要。这样可以确保文本的拆分方式与预训练语料库相同，并且在预训练期间使用相同的标记到索引的对应关系（通常称为 *词汇表*）。
</Tip>

使用 [`AutoTokenizer.from_pretrained`] 方法加载预训练的令牌化器。这将下载模型预训练时使用的 

*词汇表*：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

然后将文本传递给令牌化器：
```py
>>> encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
>>> print(encoded_input)
{'input_ids': [101, 2079, 2025, 19960, 10362, 1999, 1996, 3821, 1997, 16657, 1010, 2005, 2027, 2024, 11259, 1998, 4248, 2000, 4963, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

令牌化器返回一个包含三个重要项目的字典：

* [input_ids](glossary#input-ids) 是句子中每个标记对应的索引。
* [attention_mask](glossary#attention-mask) 指示是否应该关注标记。* [token_type_ids](glossary#token-type-ids) 用于标识一个标记属于哪个序列，当存在多个序列时。

通过解码 `input_ids` 来返回输入：
```py
>>> tokenizer.decode(encoded_input["input_ids"])
'[CLS] Do not meddle in the affairs of wizards, for they are subtle and quick to anger. [SEP]'
```

正如您所见，令牌化器添加了两个特殊标记 - `CLS` 和 `SEP`（分类器和分隔符） - 到句子中。并非所有模型都需要特殊标记，但如果需要，令牌化器会自动为您添加。
如果有多个要预处理的句子，请将它们作为列表传递给令牌化器：
```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_inputs = tokenizer(batch_sentences)
>>> print(encoded_inputs)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102], 
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102], 
               [101, 1327, 1164, 5450, 23434, 136, 102]], 
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0]], 
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1]]}
```

### Pad 填充
句子的长度并不总是相同，这可能会成为一个问题，因为张量（模型输入）需要具有统一的形状。填充是一种策略，通过向较短的句子中添加特殊的 *填充标记* 来确保张量是矩形的。

将 `padding` 参数设置为 `True`，以便将批次中较短的序列填充到与最长序列相匹配的长度：

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0], 
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102], 
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

第一句和第三句现在都被填充为 `0`，因为它们较短。

### 截断 Truncation

另一方面，有时候序列可能对模型来说太长了。在这种情况下，您需要将序列截断为较短的长度。

将 `truncation` 参数设置为 `True`，以将序列截断为模型接受的最大长度：
```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0], 
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102], 
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

<Tip>

请参阅 [填充和截断](./pad_truncation) 概念指南，以了解更多不同的填充和截断参数。

</Tip>

### 构建张量

最后，您希望令牌化器返回实际用于模型的张量。
将 `return_tensors` 参数设置为 `pt`（用于 PyTorch）或 `tf`（用于 TensorFlow）：

<frameworkcontent> 
<pt> 

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
>>> print(encoded_input)
{'input_ids': tensor([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
                      [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
                      [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]]), 
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
```
</pt> 
<tf> 

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="tf")
>>> print(encoded_input)
{'input_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
       [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
       [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
      dtype=int32)>, 
 'token_type_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>, 
 'attention_mask': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>}
```
</tf>
</frameworkcontent>



## 音频

对于音频任务，您将需要一个 [特征提取器](main_classes/feature_extractor) 来准备模型的数据集。特征提取器旨在从原始音频数据中提取特征，并将其转换为张量。

加载 [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) 数据集（有关如何加载数据集的详细信息，请参阅🤗 [数据集教程](https://huggingface.co/docs/datasets/load_hub.html)）以了解如何在音频数据集中使用特征提取器：

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

访问 `audio` 列的第一个元素以查看输入。调用 `audio` 列会自动加载和重新采样音频文件：
```py
>>> dataset[0]["audio"]
{'array': array([ 0.        ,  0.00024414, -0.00024414, ..., -0.00024414,
         0.        ,  0.        ], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'sampling_rate': 8000}
```

这将返回三个项目：

* `array` 是加载的语音信号 - 可能已经重新采样 - 作为 1D 数组。
* `path` 指向音频文件的位置。
* `sampling_rate` 是每秒测量的语音信号数据点数。

在本教程中，您将使用 [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) 模型。查看模型卡片，您将了解到 Wav2Vec2 是在 16kHz 采样的语音音频上进行预训练的。因此，您的音频数据的采样率必须与用于预训练模型的数据集的采样率相匹配。如果数据的采样率不同，则需要对数据进行重新采样。

1. 使用🤗数据集的 [`~datasets.Dataset.cast_column`] 方法将采样率提升到 16kHz：
```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
```

2. 再次调用 `audio` 列以重新采样音频文件：
```py
>>> dataset[0]["audio"]
{'array': array([ 2.3443763e-05,  2.1729663e-04,  2.2145823e-04, ...,
         3.8356509e-05, -7.3497440e-06, -2.1754686e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'sampling_rate': 16000}
```

接下来，加载特征提取器以对输入进行归一化和填充。当填充文本数据时，会添加一个 `0` 来表示较短的序列。同样的思想也适用于音频数据。特征提取器会将 `array` 中的 `0`（被解释为静默）添加进去。

使用 [`AutoFeatureExtractor.from_pretrained`] 加载特征提取器：
```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

将音频 `array` 传递给特征提取器。我们还建议在特征提取器中添加 `sampling_rate` 参数，以便更好地调试可能发生的静音错误。
```py
>>> audio_input = [dataset[0]["audio"]["array"]]
>>> feature_extractor(audio_input, sampling_rate=16000)
{'input_values': [array([ 3.8106556e-04,  2.7506407e-03,  2.8015103e-03, ...,
        5.6335266e-04,  4.6588284e-06, -1.7142107e-04], dtype=float32)]}
```

就像令牌化器一样，您可以应用填充或截断来处理批次中的可变序列。请看一下这两个音频样本的序列长度：
```py
>>> dataset[0]["audio"]["array"].shape
(173398,)

>>> dataset[1]["audio"]["array"].shape
(106496,)
```

创建一个函数来预处理数据集，以使音频样本具有相同的长度。指定最大样本长度，特征提取器将填充或截断序列以匹配该长度：
```py
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays,
...         sampling_rate=16000,
...         padding=True,
...         max_length=100000,
...         truncation=True,
...     )
...     return inputs
```

将 `preprocess_function` 应用于数据集中的前几个示例：
```py
>>> processed_dataset = preprocess_function(dataset[:5])
```

现在样本的长度是相同的，并且与指定的最大长度匹配。您现在可以将处理过的数据集传递给模型！
```py
>>> processed_dataset["input_values"][0].shape
(100000,)

>>> processed_dataset["input_values"][1].shape
(100000,)
```

## 计算机视觉

对于计算机视觉任务，您将需要一个 [图像处理器 (Image Processor)](main_classes/image_processor) 来准备您的数据集，以便用于模型。图像预处理由几个步骤组成，将图像转换为模型所期望的输入。这些步骤包括但不限于调整大小、归一化、颜色通道校正和将图像转换为张量。

<Tip>

图像预处理通常会跟随某种形式的图像增强。图像预处理和图像增强都会对图像数据进行转换，但它们的目的不同：
- 图像增强以一种可以帮助防止过拟合并增加模型的鲁棒性的方式改变图像。您可以在数据增强中进行创造性的操作-调整亮度和颜色、裁剪、旋转、调整大小、缩放等。但是，请注意不要通过增强改变图像的含义。- 图像预处理可以保证图像与模型预期的输入格式相匹配。在微调计算机视觉模型时，图像必须与模型最初训练时的预处理方式完全一致。
您可以使用任何您喜欢的图像增强库。

对于图像预处理，请使用与模型关联的 `ImageProcessor`。

</Tip>

加载 [food101](https://huggingface.co/datasets/food101) 数据集（有关如何加载数据集的更多详细信息，请参见🤗 [Datasets 教程](https://huggingface.co/docs/datasets/load_hub.html)），以了解如何在计算机视觉数据集中使用图像处理器 (Image Processor)：
<Tip>

使用🤗 Datasets 的 `split` 参数仅加载训练集的小样本，因为数据集非常大！
</Tip>
```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("food101", split="train[:100]")
```

接下来，看看带有🤗 Datasets [`Image`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=image#datasets.Image) 特征的图像：
```py
>>> dataset[0]["image"]
```

<div class="flex justify-center">    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vision-preprocess-tutorial.png"/> </div>
使用 [`AutoImageProcessor.from_pretrained`] 加载图像处理器 (Image Processor)：
```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

首先，让我们添加一些图像增强。您可以使用任何您喜欢的库，但在本教程中，我们将使用 torchvision 的 [`transforms`](https://pytorch.org/vision/stable/transforms.html) 模块。如果您想使用另一个数据增强库，请查看 [Albumentations](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_albumentations.ipynb) 或 [Kornia notebooks](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_kornia.ipynb)。

1. 这里我们使用 [`Compose`](https://pytorch.org/vision/master/generated/torchvision.transforms.Compose.html) 来链接一些转换- [`RandomResizedCrop`](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html) 和 [`ColorJitter`](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html)。请注意，对于调整大小，我们可以从 `image_processor` 中获取图像尺寸要求。

对于某些模型，期望的是精确的高度和宽度，而对于其他模型，只定义了 `shortest_edge`。2. 模型接受 [`pixel_values`](model_doc/visionencoderdecoder#transformers.VisionEncoderDecoderModel.forward.pixel_values) 作为其输入。

 `ImageProcessor` 可以负责对图像进行归一化，并生成适当的张量。width are expected, for others only the `shortest_edge` is defined.

```py
>>> from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose

>>> size = (
...     image_processor.size["shortest_edge"]
...     if "shortest_edge" in image_processor.size
...     else (image_processor.size["height"], image_processor.size["width"])
... )

>>> _transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)])
```

2. 这个模型接受 `pixel_values` 作为输入。`ImageProcessor` 可以负责对图像进行归一化和生成适当的张量。创建一个函数，将图像增强和图像预处理结合起来处理一批图像，并生成 `pixel_values`。
```py
>>> def transforms(examples):
...     images = [_transforms(img.convert("RGB")) for img in examples["image"]]
...     examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
...     return examples
```

<Tip>

在上面的示例中，我们将 `do_resize=False`，因为我们已经在图像增强转换中调整了图像的大小，并利用了适当的 `image_processor` 的 `size` 属性。如果您在图像增强过程中不调整图像的大小，请省略此参数。默认情况下，`ImageProcessor` 将处理调整大小。


如果您希望将图像归一化作为增强转换的一部分，请使用 `image_processor.image_mean` 和 `image_processor.image_std` 值。and `image_processor.image_std` values.
</Tip>

3. 然后使用🤗 Datasets 的 [`set_transform`](https://huggingface.co/docs/datasets/process.html#format-transform) 来实时应用转换：
```py
>>> dataset.set_transform(transforms)
```

4. 现在，当您访问图像时，您会注意到图像处理器 (Image Processor)已经添加了 `pixel_values`。您现在可以将处理过的数据集传递给模型！
```py
>>> dataset[0].keys()
```

在应用转换后，图像的效果如下所示。该图像已进行随机裁剪，其颜色属性也有所不同。
```py
>>> import numpy as np
>>> import matplotlib.pyplot as plt

>>> img = dataset[0]["pixel_values"]
>>> plt.imshow(img.permute(1, 2, 0))
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/preprocessed_image.png"/>
</div>
<Tip>
对于物体检测、语义分割、实例分割和全景分割等任务，`ImageProcessor` 提供了后处理方法。这些方法将模型的原始输出转换为有意义的预测，如边界框或分割图。
</Tip>

### 填充

在某些情况下，例如在微调 [DETR](./model_doc/detr) 时，模型会在训练时应用比例增强。这可能导致批次中的图像大小不同。您可以使用 [`DetrImageProcessor.pad_and_create_pixel_mask`] 从 [`DetrImageProcessor`] 中定义一个自定义的 `collate_fn` 来将图像批量处理在一起。


```py
>>> def collate_fn(batch):
...     pixel_values = [item["pixel_values"] for item in batch]
...     encoding = image_processor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
...     labels = [item["labels"] for item in batch]
...     batch = {}
...     batch["pixel_values"] = encoding["pixel_values"]
...     batch["pixel_mask"] = encoding["pixel_mask"]
...     batch["labels"] = labels
...     return batch
```

## 多模态

对于涉及多模态输入的任务，您将需要一个 [处理器](main_classes/processors) 来准备您的数据集，以便用于模型。处理器将两个处理对象（例如分词器 (Tokenizer)和特征提取器）结合在一起。

加载 [LJ Speech](https://huggingface.co/datasets/lj_speech) 数据集（有关如何加载数据集的更多详细信息，请参见🤗 [Datasets 教程](https://huggingface.co/docs/datasets/load_hub.html)），
以了解如何在自动语音识别（ASR）中使用处理器：

```py
>>> from datasets import load_dataset

>>> lj_speech = load_dataset("lj_speech", split="train")
```

对于 ASR，您主要关注的是 `audio` 和 `text`，因此可以删除其他列：
```py
>>> lj_speech = lj_speech.map(remove_columns=["file", "id", "normalized_text"])
```

现在看一下 `audio` 和 `text` 列：
```py
>>> lj_speech[0]["audio"]
{'array': array([-7.3242188e-04, -7.6293945e-04, -6.4086914e-04, ...,
         7.3242188e-04,  2.1362305e-04,  6.1035156e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/917ece08c95cf0c4115e45294e3cd0dee724a1165b7fc11798369308a465bd26/LJSpeech-1.1/wavs/LJ001-0001.wav',
 'sampling_rate': 22050}

>>> lj_speech[0]["text"]
'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition'
```

请记住，您应该始终将音频数据集的采样率 [重新采样](preprocessing#audio) 以与用于预训练模型的数据集的采样率相匹配！
```py
>>> lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=16_000))
```

使用 [`AutoProcessor.from_pretrained`] 加载处理器：
```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
```

1. 创建一个将 `array` 中的音频数据处理为 `input_values`，将 `text` 标记化为 `labels` 的函数。这些是模型的输入：
```py
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))

...     return example
```

2. 将 `prepare_dataset` 函数应用于一个样本：
```py
>>> prepare_dataset(lj_speech[0])
```

处理器现在已经添加了 `input_values` 和 `labels`，并且采样率也已经正确降采样为 16kHz。您现在可以将处理过的数据集传递给模型！