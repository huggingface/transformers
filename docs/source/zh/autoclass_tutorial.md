<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 使用AutoClass加载预训练实例

由于存在许多不同的Transformer架构，因此为您的checkpoint创建一个可用架构可能会具有挑战性。通过`AutoClass`可以自动推断并从给定的checkpoint加载正确的架构, 这也是🤗 Transformers易于使用、简单且灵活核心规则的重要一部分。`from_pretrained()`方法允许您快速加载任何架构的预训练模型，因此您不必花费时间和精力从头开始训练模型。生成这种与checkpoint无关的代码意味着，如果您的代码适用于一个checkpoint，它将适用于另一个checkpoint - 只要它们是为了类似的任务进行训练的 - 即使架构不同。

<Tip>

请记住，架构指的是模型的结构，而checkpoints是给定架构的权重。例如，[BERT](https://huggingface.co/google-bert/bert-base-uncased)是一种架构，而`google-bert/bert-base-uncased`是一个checkpoint。模型是一个通用术语，可以指代架构或checkpoint。


</Tip>

在这个教程中，学习如何：

* 加载预训练的分词器（`tokenizer`）
* 加载预训练的图像处理器(`image processor`)
* 加载预训练的特征提取器(`feature extractor`)
* 加载预训练的处理器(`processor`)
* 加载预训练的模型。


## AutoTokenizer

几乎所有的NLP任务都以`tokenizer`开始。`tokenizer`将您的输入转换为模型可以处理的格式。

使用[`AutoTokenizer.from_pretrained`]加载`tokenizer`：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
```

然后按照如下方式对输入进行分词：

```py
>>> sequence = "In a hole in the ground there lived a hobbit."
>>> print(tokenizer(sequence))
{'input_ids': [101, 1999, 1037, 4920, 1999, 1996, 2598, 2045, 2973, 1037, 7570, 10322, 4183, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

## AutoImageProcessor

对于视觉任务，`image processor`将图像处理成正确的输入格式。

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```


## AutoFeatureExtractor

对于音频任务,`feature extractor`将音频信号处理成正确的输入格式。

使用[`AutoFeatureExtractor.from_pretrained`]加载`feature extractor`：

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained(
...     "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
... )
```

## AutoProcessor

多模态任务需要一种`processor`，将两种类型的预处理工具结合起来。例如，[LayoutLMV2](model_doc/layoutlmv2)模型需要一个`image processo`来处理图像和一个`tokenizer`来处理文本；`processor`将两者结合起来。

使用[`AutoProcessor.from_pretrained`]加载`processor`：


```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
```

## AutoModel

<frameworkcontent>
<pt>

最后，`AutoModelFor`类让你可以加载给定任务的预训练模型（参见[这里](model_doc/auto)获取可用任务的完整列表）。例如，使用[`AutoModelForSequenceClassification.from_pretrained`]加载用于序列分类的模型：

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

轻松地重复使用相同的checkpoint来为不同任务加载模型架构：


```py
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

<Tip warning={true}>

对于PyTorch模型，`from_pretrained()`方法使用`torch.load()`，它内部使用已知是不安全的`pickle`。一般来说，永远不要加载来自不可信来源或可能被篡改的模型。对于托管在Hugging Face Hub上的公共模型，这种安全风险在一定程度上得到了缓解，因为每次提交都会进行[恶意软件扫描](https://huggingface.co/docs/hub/security-malware)。请参阅[Hub文档](https://huggingface.co/docs/hub/security)以了解最佳实践，例如使用GPG进行[签名提交验证](https://huggingface.co/docs/hub/security-gpg#signing-commits-with-gpg)。

TensorFlow和Flax的checkpoints不受影响，并且可以在PyTorch架构中使用`from_tf`和`from_flax`关键字参数,通过`from_pretrained`方法进行加载,来绕过此问题。

</Tip>

一般来说，我们建议使用`AutoTokenizer`类和`AutoModelFor`类来加载预训练的模型实例。这样可以确保每次加载正确的架构。在下一个[教程](preprocessing)中，学习如何使用新加载的`tokenizer`, `image processor`, `feature extractor`和`processor`对数据集进行预处理以进行微调。

</pt>
<tf>
最后，`TFAutoModelFor`类允许您加载给定任务的预训练模型（请参阅[这里](model_doc/auto)获取可用任务的完整列表）。例如，使用[`TFAutoModelForSequenceClassification.from_pretrained`]加载用于序列分类的模型：

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

轻松地重复使用相同的checkpoint来为不同任务加载模型架构：

```py
>>> from transformers import TFAutoModelForTokenClassification

>>> model = TFAutoModelForTokenClassification.from_pretrained("distilbert/distilbert-base-uncased")
```
一般来说，我们推荐使用`AutoTokenizer`类和`TFAutoModelFor`类来加载模型的预训练实例。这样可以确保每次加载正确的架构。在下一个[教程](preprocessing)中，学习如何使用新加载的`tokenizer`, `image processor`, `feature extractor`和`processor`对数据集进行预处理以进行微调。

</tf>
</frameworkcontent>
