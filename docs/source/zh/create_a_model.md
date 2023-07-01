<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证，第 2 版（“许可证”）授权；除非符合许可证，否则不得使用此文件许可证。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按“按原样”基础分发的，不附带任何明示或暗示的保证或条件。有关许可证的特定语言的权限和限制，请参阅许可证。
⚠️ 请注意，此文件是 Markdown 格式，但包含了我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确显示。
-->

# 创建自定义架构

[`AutoClass`](model_doc/auto) 会自动推断模型架构并下载预训练的配置和权重。通常，我们建议使用 `AutoClass` 来生成与检查点无关的代码。但是，希望对特定模型参数有更多控制权的用户可以从几个基类创建自定义🤗 Transformers 模型。这对于对🤗 Transformers 模型进行研究、训练或实验感兴趣的任何人都非常有用。在本指南中，深入了解如何创建没有 `AutoClass` 的自定义模型。学习如何：
- 加载和自定义模型配置。- 创建模型架构。- 为文本创建慢速和快速分词器 (Tokenizer)。- 为视觉任务创建图像处理器 (Image Processor)。- 为音频任务创建特征提取器。- 为多模态任务创建处理器。

## 配置 Configuration

[配置](main_classes/configuration) 是指模型的特定属性。每个模型配置都具有不同的属性；例如，所有 NLP 模型都具有 `hidden_size`、`num_attention_heads`、`num_hidden_layers` 和 `vocab_size` 属性。这些属性用于指定构建模型所需的注意力头或隐藏层的数量。

通过访问 [`DistilBertConfig`] 来详细了解 [DistilBERT](model_doc/distilbert) 的属性：
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

[`DistilBertConfig`] 显示了构建基本 [`DistilBertModel`] 所使用的所有默认属性。所有属性都是可自定义的，为实验提供了空间。例如，您可以使用 `activation` 参数自定义默认模型的激活函数。

- 使用 `attention_dropout` 参数，在注意力概率中使用更高的 dropout 比率。

- 使用 [`~PretrainedConfig.from_pretrained`] 函数可修改预训练模型属性：

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

预训练模型属性可在 [`~PretrainedConfig.from_pretrained`] 函数中进行修改：
```py
>>> my_config = DistilBertConfig.from_pretrained("distilbert-base-uncased", activation="relu", attention_dropout=0.4)
```

完成模型配置后，可以使用 [`~PretrainedConfig.save_pretrained`] 保存配置。配置文件将以 JSON 文件的形式存储在指定的保存目录中：
```py
>>> my_config.save_pretrained(save_directory="./your_model_save_path")
```

要重用配置文件，请使用 [`~PretrainedConfig.from_pretrained`] 加载它：
```py
>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/config.json")
```

<Tip>

您还可以将配置文件保存为字典，甚至只保存自定义配置属性与默认配置属性之间的差异！有关详细信息，请参阅 [配置](main_classes/configuration) 文档。
</Tip>

## 模型

下一步是创建一个 [模型](main_classes/models)。模型（也可以称为架构）定义每个层所做的工作和进行的操作。配置中的属性（例如 `num_hidden_layers`）用于定义架构。每个模型都共享 [`PreTrainedModel`] 基类和一些常见方法，例如调整输入嵌入和修剪自注意力头。此外，所有模型还都是 [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)、[`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) 或 [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/flax.linen.html#module) 的子类。
这意味着模型与各自框架的使用方式兼容。

<frameworkcontent> 
<pt>

 将自定义配置属性加载到模型中：
```py
>>> from transformers import DistilBertModel

>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/config.json")
>>> model = DistilBertModel(my_config)
```

这将创建一个使用随机值而不是预训练权重的模型。在训练模型之前，您无法将其用于任何有用的用途。

训练是一种昂贵且耗时的过程。

通常情况下，最好使用预训练模型以更少的资源获取更好的结果。
使用 [`~PreTrainedModel.from_pretrained`] 创建预训练模型：

```py
>>> model = DistilBertModel.from_pretrained("distilbert-base-uncased")
```

当加载预训练权重时，如果模型由🤗 Transformers 提供，将自动加载默认模型配置。但是，如果需要，仍然可以替换默认模型配置属性的一部分或全部：
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

这将创建一个使用随机值而不是预训练权重的模型。在训练模型之前，您无法将其用于任何有用的用途。训练是一种昂贵且耗时的过程。通常情况下，最好使用预训练模型以更少的资源获取更好的结果。

使用 [`~TFPreTrainedModel.from_pretrained`] 创建预训练模型：

```py
>>> tf_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
```

当加载预训练权重时，如果模型由🤗 Transformers 提供，将自动加载默认模型配置。但是，如果需要，仍然可以替换默认模型配置属性的一部分或全部：
```py
>>> tf_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased", config=my_config)
```
</tf> 
</frameworkcontent>

### Model heads 模型头

此时，您拥有一个基本的 DistilBERT 模型，它输出 *隐藏状态*。隐藏状态将作为输入传递给模型头以生成最终输出。只要模型支持任务（例如，您不能将 DistilBERT 用于翻译等序列到序列的任务），🤗 Transformers 为每个任务提供不同的模型头。

<frameworkcontent> 
<pt> 

例如，[`DistilBertForSequenceClassification`] 是一个带有序列分类头的基本 DistilBERT 模型。序列分类头是在汇聚的输出之上的线性层。

```py
>>> from transformers import DistilBertForSequenceClassification

>>> model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

通过切换到不同的模型头，可以轻松将此检查点重用于其他任务。例如，对于问答任务，您将使用 [`DistilBertForQuestionAnswering`] 模型头。问答头与序列分类头类似，只是在隐藏状态输出之上的线性层。
```py
>>> from transformers import DistilBertForQuestionAnswering

>>> model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
```
</pt> 
<tf> 

例如，[`TFDistilBertForSequenceClassification`] 是一个带有序列分类头的基本 DistilBERT 模型。序列分类头是在汇聚的输出之上的线性层。

```py
>>> from transformers import TFDistilBertForSequenceClassification

>>> tf_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

通过切换到不同的模型头，可以轻松将此检查点重用于其他任务。例如，对于问答任务，您将使用 [`TFDistilBertForQuestionAnswering`] 模型头。

问答头与序列分类头类似，只是在隐藏状态输出之上的线性层。

```py
>>> from transformers import TFDistilBertForQuestionAnswering

>>> tf_model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
```
</tf>
</frameworkcontent>

## 分词器 (Tokenizer)

在将原始文本转换为张量之前，您需要使用 [分词器 (Tokenizer)](main_classes/tokenizer) 作为文本数据的最后一个基类。🤗 Transformers 提供了两种类型的分词器 (Tokenizer)：

- [`PreTrainedTokenizer`]: 一个基于Python的分词器实现。
- [`PreTrainedTokenizerFast`]: 一个来自我们基于Rust的[🤗 Tokenizer](https://huggingface.co/docs/tokenizers/python/latest/)库的分词器。由于其采用了Rust实现，这种分词器类型在批量分词过程中具有显著的速度优势。快速分词器还提供了其他方法，如*偏移映射*，可将标记映射到其原始单词或字符。

这两种分词器都支持常见的方法，如编码和解码、添加新标记以及管理特殊标记。

<Tip warning={true}>

并非所有模型都支持快速分词器。您可以查看这个 [表格](index#supported-frameworks) 来检查模型是否支持快速分词器。

</Tip>

如果您训练了自己的分词器，可以使用您的 *词汇表* 文件创建一个分词器：

```py
>>> from transformers import DistilBertTokenizer

>>> my_tokenizer = DistilBertTokenizer(vocab_file="my_vocab_file.txt", do_lower_case=False, padding_side="left")
```

It is important to remember the vocabulary from a custom tokenizer will be different from the vocabulary generated by a pretrained model's tokenizer. You need to use a pretrained model's vocabulary if you are using a pretrained model, otherwise the inputs won't make sense. Create a tokenizer with a pretrained model's vocabulary with the [`DistilBertTokenizer`] class:

```py
>>> from transformers import DistilBertTokenizer

>>> slow_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
```

Create a fast tokenizer with the [`DistilBertTokenizerFast`] class:

```py
>>> from transformers import DistilBertTokenizerFast

>>> fast_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
```

<Tip>

By default, [`AutoTokenizer`] will try to load a fast tokenizer. You can disable this behavior by setting `use_fast=False` in `from_pretrained`.

</Tip>

## 图像处理器 (Image Processor) 

图像处理器 (Image Processor)用于处理视觉输入。它继承自基类 [`~image_processing_utils.ImageProcessingMixin`]。

要使用图像处理器 (Image Processor)，可以创建一个与您正在使用的模型关联的图像处理器 (Image Processor)。例如，如果您正在使用 [ViT](model_doc/vit) 进行图像分类，可以创建一个默认的 [`ViTImageProcessor`]：

```py
>>> from transformers import ViTImageProcessor

>>> vit_extractor = ViTImageProcessor()
>>> print(vit_extractor)
ViTImageProcessor {
  "do_normalize": true,
  "do_resize": true,
  "feature_extractor_type": "ViTImageProcessor",
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

如果您不需要进行任何自定义，只需使用 `from_pretrained` 方法加载模型的默认图像处理器参数即可。

</Tip>

修改任何 [`ViTImageProcessor`] 参数以创建您的自定义图像处理器：

```py
>>> from transformers import ViTImageProcessor

>>> my_vit_extractor = ViTImageProcessor(resample="PIL.Image.BOX", do_normalize=False, image_mean=[0.3, 0.3, 0.3])
>>> print(my_vit_extractor)
ViTImageProcessor {
  "do_normalize": false,
  "do_resize": true,
  "feature_extractor_type": "ViTImageProcessor",
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

## 特征提取器 （Feature Extractor）

特征提取器用于处理音频输入。它继承自基类 [`~feature_extraction_utils.FeatureExtractionMixin`]，并且可能还继承自 [`SequenceFeatureExtractor`] 类来处理音频输入。

要使用特征提取器，可以创建一个与您正在使用的模型关联的特征提取器。例如，如果您正在使用 [Wav2Vec2](model_doc/wav2vec2) 进行音频分类，可以创建一个默认的 [`Wav2Vec2FeatureExtractor`]：

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

如果您不需要进行任何自定义，只需使用 `from_pretrained` 方法加载模型的默认特征提取器参数即可。

</Tip>

修改任何 [`Wav2Vec2FeatureExtractor`] 的参数以进行自定义设置：

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


##  处理器（Processor）

对于支持多模态任务的模型，🤗 Transformers 提供了一个处理器类，它可以方便地将特征提取器和分词器等处理类封装成一个单一对象。例如，让我们使用 [`Wav2Vec2Processor`] 来处理自动语音识别任务（ASR）。

ASR 将音频转录为文本，因此您将需要一个特征提取器和一个分词器。

创建一个特征提取器来处理音频输入：

```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> feature_extractor = Wav2Vec2FeatureExtractor(padding_value=1.0, do_normalize=True)
```

Create a tokenizer to handle the text inputs:

```py
>>> from transformers import Wav2Vec2CTCTokenizer

>>> tokenizer = Wav2Vec2CTCTokenizer(vocab_file="my_vocab_file.txt")
```

Combine the feature extractor and tokenizer in [`Wav2Vec2Processor`]:

```py
>>> from transformers import Wav2Vec2Processor

>>> processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
```

通过配置类、模型类和额外的预处理类（分词器、图像处理器、特征提取器或处理器），您可以创建🤗 Transformers 支持的任何模型。每个基类都是可配置的，允许您使用所需的特定属性。您可以轻松地设置一个用于训练的模型，或修改一个现有的预训练模型进行微调。
