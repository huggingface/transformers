<!--版权 2023 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的保证或条件。请参阅许可证以了解特定语言下的权限和限制。请注意，此文件是 Markdown 格式，但包含特定于我们的 doc-builder（类似于 MDX）的语法，可能无法在 Markdown 查看器中正确
渲染。-->


# 零样本图像分类

[[在 Colab 中打开]]

零样本图像分类是一项任务，涉及使用未经过标记示例数据训练的模型将图像分类到不同的类别中。传统上，图像分类需要在特定的一组带标签图像上训练模型，该模型学习将某些图像特征映射到标签。

当需要在这样的模型上使用分类任务引入一组新标签时，需要进行微调以“重新校准”模型。相比之下，零样本或开放词汇图像分类模型通常是多模态模型，已在大量图像和相关描述的数据集上进行了训练。

这些模型学习了对齐的视觉语言表示，可以用于许多下游任务，包括零样本图像分类。new set of labels, fine-tuning is required to "recalibrate" the model.

这是一种更灵活的图像分类方法，允许模型推广到新的和未知的类别，而无需额外的训练数据，并允许用户使用目标对象的自由形式文本描述查询图像。
在本指南中，您将学习如何：

* 创建零样本图像分类流水线
* 手动运行零样本图像分类推理

开始之前，请确保您已安装所有必要的库：


```bash
pip install -q transformers
```

## 零样本图像分类流水线

尝试使用支持零样本图像分类的模型进行推理的最简单方法是使用相应的 [`pipeline`]。
从 [Hugging Face Hub 的检查点](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&sort=downloads) 实例化一个流水线：

```python
>>> from transformers import pipeline

>>> checkpoint = "openai/clip-vit-large-patch14"
>>> detector = pipeline(model=checkpoint, task="zero-shot-image-classification")
```

接下来，选择要分类的图像。
```py
>>> from PIL import Image
>>> import requests

>>> url = "https://unsplash.com/photos/g8oS8-82DxI/download?ixid=MnwxMjA3fDB8MXx0b3BpY3x8SnBnNktpZGwtSGt8fHx8fDJ8fDE2NzgxMDYwODc&force=true&w=640"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image
```

<div class="flex justify-center">     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/owl.jpg" alt="一只猫头鹰的照片"/> </div>

将图像和候选对象标签传递给流水线。

在这里，我们直接传递图像；其他合适的选项包括图像的本地路径或图像网址。候选标签可以是简单的单词，就像这个例子中一样，也可以更具描述性。

```py
>>> predictions = classifier(image, candidate_labels=["fox", "bear", "seagull", "owl"])
>>> predictions
[{'score': 0.9996670484542847, 'label': 'owl'},
 {'score': 0.000199399160919711, 'label': 'seagull'},
 {'score': 7.392891711788252e-05, 'label': 'fox'},
 {'score': 5.96074532950297e-05, 'label': 'bear'}]
```

## 手动进行零样本图像分类

既然您已经了解了如何使用零样本图像分类流水线，让我们看看如何手动运行零样本图像分类。

首先，从 [Hugging Face Hub 的检查点](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&sort=downloads) 加载模型和相关的处理器。


这里我们将使用与之前相同的检查点：

```py
>>> from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

>>> model = AutoModelForZeroShotImageClassification.from_pretrained(checkpoint)
>>> processor = AutoProcessor.from_pretrained(checkpoint)
```

让我们选择一张不同的图像来改变一下。
```py
>>> from PIL import Image
>>> import requests

>>> url = "https://unsplash.com/photos/xBRQfR2bqNI/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjc4Mzg4ODEx&force=true&w=640"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image
```

<div class="flex justify-center">     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg" alt="一辆汽车的照片"/> </div>

使用处理器准备模型的输入。处理器组合了一个图像处理器 (Image Processor)，通过调整和归一化图像来为模型准备图像，以及一个标记器，负责文本输入。

```py
>>> candidate_labels = ["tree", "car", "bike", "cat"]
>>> inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)
```

将输入通过模型，并对结果进行后处理：
```py
>>> import torch

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits = outputs.logits_per_image[0]
>>> probs = logits.softmax(dim=-1).numpy()
>>> scores = probs.tolist()

>>> result = [
...     {"score": score, "label": candidate_label}
...     for score, candidate_label in sorted(zip(probs, candidate_labels), key=lambda x: -x[0])
... ]

>>> result
[{'score': 0.998572, 'label': 'car'},
 {'score': 0.0010570387, 'label': 'bike'},
 {'score': 0.0003393686, 'label': 'tree'},
 {'score': 3.1572064e-05, 'label': 'cat'}]
```