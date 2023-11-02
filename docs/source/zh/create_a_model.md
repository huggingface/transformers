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

# 创建自定义架构

[`AutoClass`](model_doc/auto) 自动推断模型架构并下载预训练的配置和权重。一般来说，我们建议使用 `AutoClass` 生成与检查点（checkpoint）无关的代码。希望对特定模型参数有更多控制的用户，可以仅从几个基类创建自定义的 🤗 Transformers 模型。这对于任何有兴趣学习、训练或试验 🤗 Transformers 模型的人可能特别有用。通过本指南，深入了解如何不通过 `AutoClass` 创建自定义模型。了解如何：

- 加载并自定义模型配置。
- 创建模型架构。
- 为文本创建慢速和快速分词器。
- 为视觉任务创建图像处理器。
- 为音频任务创建特征提取器。
- 为多模态任务创建处理器。

## 配置

[配置](main_classes/configuration) 涉及到模型的具体属性。每个模型配置都有不同的属性；例如，所有 NLP 模型都共享 `hidden_size`、`num_attention_heads`、 `num_hidden_layers` 和 `vocab_size` 属性。这些属性用于指定构建模型时的注意力头数量或隐藏层层数。

访问 [`DistilBertConfig`] 以更近一步了解 [DistilBERT](model_doc/distilbert)，检查它的属性：

```py
>>> from transformers import DistilBertConfig

>>> config = DistilBertConfig()
>>> print(config)
DistilBertConfig {
  "activation": "gelu",
  "attention_dropout": 0.1,
  "dim": 768,
  "dropout": 0.1,
  "hidden_dim": 3072,
  "initializer_range": 0.02,
  "max_position_embeddings": 512,
  "model_type": "distilbert",
  "n_heads": 12,
  "n_layers": 6,
  "pad_token_id": 0,
  "qa_dropout": 0.1,
  "seq_classif_dropout": 0.2,
  "sinusoidal_pos_embds": false,
  "transformers_version": "4.16.2",
  "vocab_size": 30522
}
```

[`DistilBertConfig`] 显示了构建基础 [`DistilBertModel`] 所使用的所有默认属性。所有属性都可以进行自定义，为实验创造了空间。例如，您可以将默认模型自定义为：

- 使用 `activation` 参数尝试不同的激活函数。
- 使用 `attention_dropout` 参数为 attention probabilities 使用更高的 dropout ratio。

```py
>>> my_config = DistilBertConfig(activation="relu", attention_dropout=0.4)
>>> print(my_config)
DistilBertConfig {
  "activation": "relu",
  "attention_dropout": 0.4,
  "dim": 768,
  "dropout": 0.1,
  "hidden_dim": 3072,
  "initializer_range": 0.02,
  "max_position_embeddings": 512,
  "model_type": "distilbert",
  "n_heads": 12,
  "n_layers": 6,
  "pad_token_id": 0,
  "qa_dropout": 0.1,
  "seq_classif_dropout": 0.2,
  "sinusoidal_pos_embds": false,
  "transformers_version": "4.16.2",
  "vocab_size": 30522
}
```

预训练模型的属性可以在 [`~PretrainedConfig.from_pretrained`] 函数中进行修改：

```py
>>> my_config = DistilBertConfig.from_pretrained("distilbert-base-uncased", activation="relu", attention_dropout=0.4)
```

当你对模型配置满意时，可以使用 [`~PretrainedConfig.save_pretrained`] 来保存配置。你的配置文件将以 JSON 文件的形式存储在指定的保存目录中：

```py
>>> my_config.save_pretrained(save_directory="./your_model_save_path")
```

要重用配置文件，请使用 [`~PretrainedConfig.from_pretrained`] 进行加载：

```py
>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/config.json")
```

<Tip>

你还可以将配置文件保存为字典，甚至只保存自定义配置属性与默认配置属性之间的差异！有关更多详细信息，请参阅 [配置](main_classes/configuration) 文档。

</Tip>

## 模型

接下来，创建一个[模型](main_classes/models)。模型，也可泛指架构，定义了每一层网络的行为以及进行的操作。配置中的 `num_hidden_layers` 等属性用于定义架构。每个模型都共享基类 [`PreTrainedModel`] 和一些常用方法，例如调整输入嵌入的大小和修剪自注意力头。此外，所有模型都是 [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)、[`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) 或 [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/flax.linen.html#module) 的子类。这意味着模型与各自框架的用法兼容。

<frameworkcontent>
<pt>
将自定义配置属性加载到模型中：

```py
>>> from transformers import DistilBertModel

>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/config.json")
>>> model = DistilBertModel(my_config)
```

这段代码创建了一个具有随机参数而不是预训练权重的模型。在训练该模型之前，您还无法将该模型用于任何用途。训练是一项昂贵且耗时的过程。通常来说，最好使用预训练模型来更快地获得更好的结果，同时仅使用训练所需资源的一小部分。

使用 [`~PreTrainedModel.from_pretrained`] 创建预训练模型：

```py
>>> model = DistilBertModel.from_pretrained("distilbert-base-uncased")
```

当加载预训练权重时，如果模型是由 🤗 Transformers 提供的，将自动加载默认模型配置。然而，如果你愿意，仍然可以将默认模型配置的某些或者所有属性替换成你自己的配置：

```py
>>> model = DistilBertModel.from_pretrained("distilbert-base-uncased", config=my_config)
```
</pt>
<tf>
将自定义配置属性加载到模型中：

```py
>>> from transformers import TFDistilBertModel

>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/my_config.json")
>>> tf_model = TFDistilBertModel(my_config)
```

这段代码创建了一个具有随机参数而不是预训练权重的模型。在训练该模型之前，您还无法将该模型用于任何用途。训练是一项昂贵且耗时的过程。通常来说，最好使用预训练模型来更快地获得更好的结果，同时仅使用训练所需资源的一小部分。

使用 [`~TFPreTrainedModel.from_pretrained`] 创建预训练模型：

```py
>>> tf_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
```

当加载预训练权重时，如果模型是由 🤗 Transformers 提供的，将自动加载默认模型配置。然而，如果你愿意，仍然可以将默认模型配置的某些或者所有属性替换成自己的配置：

```py
>>> tf_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased", config=my_config)
```
</tf>
</frameworkcontent>

### 模型头（Model heads）

此时，你已经有了一个输出*隐藏状态*的基础 DistilBERT 模型。隐藏状态作为输入传递到模型头以生成最终输出。🤗 Transformers 为每个任务提供不同的模型头，只要模型支持该任务（即，您不能使用 DistilBERT 来执行像翻译这样的序列到序列任务）。

<frameworkcontent>
<pt>
例如，[`DistilBertForSequenceClassification`] 是一个带有序列分类头（sequence classification head）的基础 DistilBERT 模型。序列分类头是池化输出之上的线性层。

```py
>>> from transformers import DistilBertForSequenceClassification

>>> model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

通过切换到不同的模型头，可以轻松地将此检查点重复用于其他任务。对于问答任务，你可以使用 [`DistilBertForQuestionAnswering`] 模型头。问答头（question answering head）与序列分类头类似，不同点在于它是隐藏状态输出之上的线性层。

```py
>>> from transformers import DistilBertForQuestionAnswering

>>> model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
```
</pt>
<tf>
例如，[`TFDistilBertForSequenceClassification`] 是一个带有序列分类头（sequence classification head）的基础 DistilBERT 模型。序列分类头是池化输出之上的线性层。

```py
>>> from transformers import TFDistilBertForSequenceClassification

>>> tf_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

通过切换到不同的模型头,可以轻松地将此检查点重复用于其他任务。对于问答任务，你可以使用 [`TFDistilBertForQuestionAnswering`] 模型头。问答头（question answering head）与序列分类头类似，不同点在于它是隐藏状态输出之上的线性层。

```py
>>> from transformers import TFDistilBertForQuestionAnswering

>>> tf_model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
```
</tf>
</frameworkcontent>

## 分词器

在将模型用于文本数据之前，你需要的最后一个基类是 [tokenizer](main_classes/tokenizer)，它用于将原始文本转换为张量。🤗 Transformers 支持两种类型的分词器：

- [`PreTrainedTokenizer`]：分词器的Python实现
- [`PreTrainedTokenizerFast`]：来自我们基于 Rust 的 [🤗 Tokenizer](https://huggingface.co/docs/tokenizers/python/latest/) 库的分词器。因为其使用了 Rust 实现，这种分词器类型的速度要快得多，尤其是在批量分词（batch tokenization）的时候。快速分词器还提供其他的方法，例如*偏移映射（offset mapping）*，它将标记（token）映射到其原始单词或字符。

这两种分词器都支持常用的方法，如编码和解码、添加新标记以及管理特殊标记。

<Tip warning={true}>

并非每个模型都支持快速分词器。参照这张 [表格](index#supported-frameworks) 查看模型是否支持快速分词器。

</Tip>

如果您训练了自己的分词器，则可以从*词表*文件创建一个分词器：

```py
>>> from transformers import DistilBertTokenizer

>>> my_tokenizer = DistilBertTokenizer(vocab_file="my_vocab_file.txt", do_lower_case=False, padding_side="left")
```

请务必记住，自定义分词器生成的词表与预训练模型分词器生成的词表是不同的。如果使用预训练模型，则需要使用预训练模型的词表，否则输入将没有意义。 使用 [`DistilBertTokenizer`] 类创建具有预训练模型词表的分词器：

```py
>>> from transformers import DistilBertTokenizer

>>> slow_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
```

使用 [`DistilBertTokenizerFast`] 类创建快速分词器：

```py
>>> from transformers import DistilBertTokenizerFast

>>> fast_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
```

<Tip>

默认情况下，[`AutoTokenizer`] 将尝试加载快速标记生成器。你可以通过在 `from_pretrained` 中设置 `use_fast=False` 以禁用此行为。

</Tip>

## 图像处理器

图像处理器用于处理视觉输入。它继承自 [`~image_processing_utils.ImageProcessingMixin`] 基类。

要使用它，需要创建一个与你使用的模型关联的图像处理器。例如，如果你使用 [ViT](model_doc/vit) 进行图像分类，可以创建一个默认的 [`ViTImageProcessor`]：

```py
>>> from transformers import ViTImageProcessor

>>> vit_extractor = ViTImageProcessor()
>>> print(vit_extractor)
ViTImageProcessor {
  "do_normalize": true,
  "do_resize": true,
  "image_processor_type": "ViTImageProcessor",
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "resample": 2,
  "size": 224
}
```

<Tip>

如果您不需要进行任何自定义，只需使用 `from_pretrained` 方法加载模型的默认图像处理器参数。

</Tip>

修改任何 [`ViTImageProcessor`] 参数以创建自定义图像处理器：

```py
>>> from transformers import ViTImageProcessor

>>> my_vit_extractor = ViTImageProcessor(resample="PIL.Image.BOX", do_normalize=False, image_mean=[0.3, 0.3, 0.3])
>>> print(my_vit_extractor)
ViTImageProcessor {
  "do_normalize": false,
  "do_resize": true,
  "image_processor_type": "ViTImageProcessor",
  "image_mean": [
    0.3,
    0.3,
    0.3
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "resample": "PIL.Image.BOX",
  "size": 224
}
```

## 特征提取器

特征提取器用于处理音频输入。它继承自 [`~feature_extraction_utils.FeatureExtractionMixin`] 基类，亦可继承 [`SequenceFeatureExtractor`] 类来处理音频输入。

要使用它，创建一个与你使用的模型关联的特征提取器。例如，如果你使用 [Wav2Vec2](model_doc/wav2vec2) 进行音频分类，可以创建一个默认的 [`Wav2Vec2FeatureExtractor`]：

```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> w2v2_extractor = Wav2Vec2FeatureExtractor()
>>> print(w2v2_extractor)
Wav2Vec2FeatureExtractor {
  "do_normalize": true,
  "feature_extractor_type": "Wav2Vec2FeatureExtractor",
  "feature_size": 1,
  "padding_side": "right",
  "padding_value": 0.0,
  "return_attention_mask": false,
  "sampling_rate": 16000
}
```

<Tip>

如果您不需要进行任何自定义，只需使用 `from_pretrained` 方法加载模型的默认特征提取器参数。

</Tip>

修改任何 [`Wav2Vec2FeatureExtractor`] 参数以创建自定义特征提取器：

```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> w2v2_extractor = Wav2Vec2FeatureExtractor(sampling_rate=8000, do_normalize=False)
>>> print(w2v2_extractor)
Wav2Vec2FeatureExtractor {
  "do_normalize": false,
  "feature_extractor_type": "Wav2Vec2FeatureExtractor",
  "feature_size": 1,
  "padding_side": "right",
  "padding_value": 0.0,
  "return_attention_mask": false,
  "sampling_rate": 8000
}
```


## 处理器

对于支持多模式任务的模型，🤗 Transformers 提供了一个处理器类，可以方便地将特征提取器和分词器等处理类包装到单个对象中。例如，让我们使用 [`Wav2Vec2Processor`] 来执行自动语音识别任务 (ASR)。 ASR 将音频转录为文本，因此您将需要一个特征提取器和一个分词器。

创建一个特征提取器来处理音频输入：

```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> feature_extractor = Wav2Vec2FeatureExtractor(padding_value=1.0, do_normalize=True)
```

创建一个分词器来处理文本输入：

```py
>>> from transformers import Wav2Vec2CTCTokenizer

>>> tokenizer = Wav2Vec2CTCTokenizer(vocab_file="my_vocab_file.txt")
```

将特征提取器和分词器合并到 [`Wav2Vec2Processor`] 中：

```py
>>> from transformers import Wav2Vec2Processor

>>> processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
```

通过两个基类 - 配置类和模型类 - 以及一个附加的预处理类（分词器、图像处理器、特征提取器或处理器），你可以创建 🤗 Transformers 支持的任何模型。 每个基类都是可配置的，允许你使用所需的特定属性。 你可以轻松设置模型进行训练或修改现有的预训练模型进行微调。
