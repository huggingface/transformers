<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 预处理

[[在Colab中打开]]

在您可以在数据集上训练模型之前，数据需要被预处理为期望的模型输入格式。无论您的数据是文本、图像还是音频，它们都需要被转换并组装成批量的张量。🤗 Transformers 提供了一组预处理类来帮助准备您的数据以供模型使用。在本教程中，您将了解以下内容：

* 对于文本，使用[分词器](./main_classes/tokenizer)将文本转换为一系列标记，创建标记的数字表示，并将它们组装成张量。
* 对于语音和音频，使用[特征提取器](./main_classes/feature_extractor)从音频波形中提取顺序特征并将其转换为张量。
* 图像输入使用[图像处理器](./main_classes/image)将图像转换为张量。
* 多模态输入，使用[处理器](./main_classes/processors)结合了分词器和特征提取器或图像处理器。

<Tip>

`AutoProcessor` **始终**有效，并自动选择适用于您使用的模型的正确类别，无论您使用的是分词器、图像处理器、特征提取器还是处理器。

</Tip>

在开始之前，请安装🤗 Datasets，以便您可以加载一些数据集来进行实验：


```bash
pip install datasets
```

## 自然语言处理

<Youtube id="Yffk5aydLzg"/>

处理文本数据的主要工具是[分词器](main_classes/tokenizer)。分词器根据一组规则将文本拆分为*标记*。然后将这些标记转换为数字，然后转换为张量，成为模型的输入。模型所需的任何附加输入都由分词器添加。

<Tip>

如果您计划使用预训练模型，重要的是使用与之关联的预训练分词器。这确保文本的拆分方式与预训练语料库相同，并在预训练期间使用相同的标记到索引的对应关系（通常称为*词汇表*）。

</Tip>

开始使用[`AutoTokenizer.from_pretrained`]方法加载一个预训练分词器。这将下载模型预训练的*词汇表*：


```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

然后将您的文本传递给分词器：


```py
>>> encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
>>> print(encoded_input)
{'input_ids': [101, 2079, 2025, 19960, 10362, 1999, 1996, 3821, 1997, 16657, 1010, 2005, 2027, 2024, 11259, 1998, 4248, 2000, 4963, 1012, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

分词器返回一个包含三个重要项的字典：

* [input_ids](glossary#input-ids) 是与句子中每个标记对应的索引。
* [attention_mask](glossary#attention-mask) 指示是否应该关注一个标记。
* [token_type_ids](glossary#token-type-ids) 在存在多个序列时标识一个标记属于哪个序列。

通过解码 `input_ids` 来返回您的输入：


```py
>>> tokenizer.decode(encoded_input["input_ids"])
'[CLS] Do not meddle in the affairs of wizards, for they are subtle and quick to anger. [SEP]'
```

如您所见，分词器向句子中添加了两个特殊标记 - `CLS` 和 `SEP`（分类器和分隔符）。并非所有模型都需要特殊标记，但如果需要，分词器会自动为您添加。

如果有多个句子需要预处理，将它们作为列表传递给分词器：


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

### 填充

句子的长度并不总是相同，这可能会成为一个问题，因为张量，即模型输入，需要具有统一的形状。填充是一种策略，通过在较短的句子中添加一个特殊的*填充标记*，以确保张量是矩形的。

将 `padding` 参数设置为 `True`，以使批次中较短的序列填充到与最长序列相匹配的长度：

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

第一句和第三句现在被填充为`0`，因为它们较短。

### 截断

在光谱的另一端，有时候一个序列可能对模型来说太长了。在这种情况下，您需要将序列截断为更短的长度。

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

查看[填充和截断](./pad_truncation)概念指南，了解更多有关不同填充和截断参数的信息。

</Tip>

### 构建张量

最后，您希望分词器返回实际馈送到模型的张量。

将 `return_tensors` 参数设置为 `pt`（对于PyTorch）或 `tf`（对于TensorFlow）：

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

对于音频任务，您需要一个[特征提取器](main_classes/feature_extractor)来准备您的数据集以供模型使用。特征提取器旨在从原始音频数据中提取特征，并将它们转换为张量。

加载[MInDS-14](https://huggingface.co/datasets/PolyAI/minds14)数据集（有关如何加载数据集的更多详细信息，请参阅🤗 [Datasets教程](https://huggingface.co/docs/datasets/load_hub.html)）以了解如何在音频数据集中使用特征提取器：


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

这会返回三个项目：

* `array` 是加载的语音信号 - 并在必要时重新采样 - 作为1D数组。
* `path` 指向音频文件的位置。
* `sampling_rate` 是每秒测量的语音信号数据点数量。

对于本教程，您将使用[Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base)模型。查看模型卡片，您将了解到Wav2Vec2是在16kHz采样的语音音频上预训练的。重要的是，您的音频数据的采样率要与用于预训练模型的数据集的采样率匹配。如果您的数据的采样率不同，那么您需要对数据进行重新采样。

1. 使用🤗 Datasets的[`~datasets.Dataset.cast_column`]方法将采样率提升到16kHz：


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

接下来，加载一个特征提取器以对输入进行标准化和填充。当填充文本数据时，会为较短的序列添加 `0`。相同的理念适用于音频数据。特征提取器向 `array` 添加 `0` - 被解释为静音。

使用 [`AutoFeatureExtractor.from_pretrained`] 加载特征提取器：


```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

将音频 `array` 传递给特征提取器。我们还建议在特征提取器中添加 `sampling_rate` 参数，以更好地调试可能发生的静默错误：


```py
>>> audio_input = [dataset[0]["audio"]["array"]]
>>> feature_extractor(audio_input, sampling_rate=16000)
{'input_values': [array([ 3.8106556e-04,  2.7506407e-03,  2.8015103e-03, ...,
        5.6335266e-04,  4.6588284e-06, -1.7142107e-04], dtype=float32)]}
```

就像分词器一样，您可以应用填充或截断来处理批次中的可变序列。查看这两个音频样本的序列长度：


```py
>>> dataset[0]["audio"]["array"].shape
(173398,)

>>> dataset[1]["audio"]["array"].shape
(106496,)
```

创建一个函数来预处理数据集，以使音频样本具有相同的长度。指定最大样本长度，特征提取器将填充或截断序列以使其匹配：


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

将`preprocess_function`应用于数据集中的前几个示例：


```py
>>> processed_dataset = preprocess_function(dataset[:5])
```

现在样本长度是相同的并与指定的最大长度匹配。您现在可以将经过处理的数据集传递给模型了！


```py
>>> processed_dataset["input_values"][0].shape
(100000,)

>>> processed_dataset["input_values"][1].shape
(100000,)
```

## 计算机视觉

对于计算机视觉任务，您需要一个[图像处理器](main_classes/image_processor)来准备数据集以供模型使用。图像预处理包括将图像转换为模型所期望的输入格式的多个步骤。这些步骤包括但不限于调整大小、标准化、颜色通道校正以及将图像转换为张量。

<Tip>

图像预处理通常遵循某种形式的图像增强。图像预处理和图像增强都会改变图像数据，但它们有不同的目的：

* 图像增强以一种方式改变图像，可以帮助防止过拟合并增加模型的鲁棒性。您可以在数据增强方面充分发挥创造性 - 调整亮度和颜色、裁剪、旋转、调整大小、缩放等。但要注意不要改变图像的含义。
* 图像预处理确保图像与模型预期的输入格式匹配。在微调计算机视觉模型时，必须对图像进行与模型最初训练时相同的预处理。

您可以使用任何您喜欢的图像增强库。对于图像预处理，请使用与模型相关联的`ImageProcessor`。

</Tip>

加载[food101](https://huggingface.co/datasets/food101)数据集（有关如何加载数据集的更多详细信息，请参阅🤗 [Datasets教程](https://huggingface.co/docs/datasets/load_hub.html)）以了解如何在计算机视觉数据集中使用图像处理器：

<Tip>

使用🤗 Datasets的`split`参数只加载训练集中的一个小样本，因为数据集相当大！

</Tip>


```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("food101", split="train[:100]")
```

接下来，使用🤗 Datasets的[`Image`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=image#datasets.Image)功能查看图像：


```py
>>> dataset[0]["image"]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vision-preprocess-tutorial.png"/>
</div>

使用 [`AutoImageProcessor.from_pretrained`] 加载图像处理器：

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

首先，让我们添加一些图像增强。您可以使用任何您喜欢的库，但在本教程中，我们将使用torchvision的[`transforms`](https://pytorch.org/vision/stable/transforms.html)模块。如果您有兴趣使用其他数据增强库，请参阅[Albumentations](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_albumentations.ipynb)或[Kornia notebooks](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_kornia.ipynb)中的示例。

1. 在这里，我们使用[`Compose`](https://pytorch.org/vision/master/generated/torchvision.transforms.Compose.html)将几个变换链接在一起 - [`RandomResizedCrop`](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html) 和 [`ColorJitter`](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html)。请注意，对于调整大小，我们可以从`image_processor`中获取图像尺寸要求。对于一些模型，期望精确的高度和宽度，对于其他模型只定义了`shortest_edge`。


```py
>>> from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose

>>> size = (
...     image_processor.size["shortest_edge"]
...     if "shortest_edge" in image_processor.size
...     else (image_processor.size["height"], image_processor.size["width"])
... )

>>> _transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)])
```

2. 模型接受 [`pixel_values`](model_doc/visionencoderdecoder#transformers.VisionEncoderDecoderModel.forward.pixel_values) 作为输入。`ImageProcessor` 可以处理图像的标准化，并生成适当的张量。创建一个函数，将图像增强和图像预处理组合为图像批处理，并生成 `pixel_values`：


```py
>>> def transforms(examples):
...     images = [_transforms(img.convert("RGB")) for img in examples["image"]]
...     examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
...     return examples
```

<Tip>

在上面的示例中，我们设置`do_resize=False`，因为我们已经在图像增强转换中调整了图像的大小，并利用了适当的`image_processor`的`size`属性。如果您在图像增强期间不调整图像的大小，请将此参数排除在外。默认情况下，`ImageProcessor`将处理调整大小。

如果希望将图像标准化为增强转换的一部分，请使用`image_processor.image_mean`和`image_processor.image_std`的值。

</Tip>

3. 然后使用🤗 Datasets的[`set_transform`](https://huggingface.co/docs/datasets/process.html#format-transform)在运行时应用这些变换：


```py
>>> dataset.set_transform(transforms)
```

4. 现在，当您访问图像时，您将注意到图像处理器已添加了 `pixel_values`。您现在可以将经过处理的数据集传递给模型了！


```py
>>> dataset[0].keys()
```

这是在应用变换后的图像样子。图像已被随机裁剪，并其颜色属性发生了变化。


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

对于诸如目标检测、语义分割、实例分割和全景分割等任务，`ImageProcessor`提供了后处理方法。这些方法将模型的原始输出转换为有意义的预测，如边界框或分割地图。

<Tip>

在某些情况下，例如，在微调[DETR](./model_doc/detr)时，模型在训练时应用尺度增强。这可能导致批处理中的图像大小不同。您可以使用[`DetrImageProcessor.pad`]来定义自定义的`collate_fn`将图像批处理在一起。

</Tip>


```py
>>> def collate_fn(batch):
...     pixel_values = [item["pixel_values"] for item in batch]
...     encoding = image_processor.pad(pixel_values, return_tensors="pt")
...     labels = [item["labels"] for item in batch]
...     batch = {}
...     batch["pixel_values"] = encoding["pixel_values"]
...     batch["pixel_mask"] = encoding["pixel_mask"]
...     batch["labels"] = labels
...     return batch
```

## 多模态

对于涉及多模态输入的任务，您需要一个[处理器](main_classes/processors)来准备数据集以适应模型。处理器将两个处理对象，如分词器和特征提取器，配对在一起。

加载[LJ Speech](https://huggingface.co/datasets/lj_speech)数据集（有关如何加载数据集的更多详细信息，请参阅🤗 [Datasets 教程](https://huggingface.co/docs/datasets/load_hub.html)）以了解如何使用处理器进行自动语音识别（ASR）：


```py
>>> from datasets import load_dataset

>>> lj_speech = load_dataset("lj_speech", split="train")
```

对于ASR（自动语音识别），主要关注`audio`和`text`，因此可以删除其他列：


```py
>>> lj_speech = lj_speech.map(remove_columns=["file", "id", "normalized_text"])
```

现在查看`audio`和`text`列：

```py
>>> lj_speech[0]["audio"]
{'array': array([-7.3242188e-04, -7.6293945e-04, -6.4086914e-04, ...,
         7.3242188e-04,  2.1362305e-04,  6.1035156e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/917ece08c95cf0c4115e45294e3cd0dee724a1165b7fc11798369308a465bd26/LJSpeech-1.1/wavs/LJ001-0001.wav',
 'sampling_rate': 22050}

>>> lj_speech[0]["text"]
'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition'
```

请记住，您应始终[重新采样](preprocessing#audio)音频数据集的采样率，以匹配用于预训练模型的数据集的采样率！


```py
>>> lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=16_000))
```

使用[`AutoProcessor.from_pretrained`]加载一个处理器：


```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
```

1. 创建一个函数，用于将包含在 array 中的音频数据处理为 input_values，并将 text 标记为 labels。这些是输入模型的数据：

```py
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))

...     return example
```

2. 将 prepare_dataset 函数应用于一个示例：

```py
>>> prepare_dataset(lj_speech[0])
```

处理器现在已经添加了 input_values 和 labels，并且采样率也正确降采样为16kHz。现在可以将处理后的数据集传递给模型！
