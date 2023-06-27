<!--版权所有 2022 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）的规定，您只能在遵守许可证的情况下使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的担保或条件。请参阅许可证以了解具体语言下的权限和限制。an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
⚠️ 特别提示：此文件是 Markdown 格式的，但包含了我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确显示。
请注意渲染。
-->
# 使用 AutoClass 加载预训练实例
由于有许多不同的 Transformer 架构，为您的检查点创建一个架构可能具有挑战性。作为🤗 Transformers 核心理念的一部分，使库易于使用、简单灵活，`AutoClass` 可以自动推断并从给定的检查点加载正确的架构。`from_pretrained()` 方法使您能够快速加载任何架构的预训练模型，因此您无需花费时间和资源从头开始训练模型。这种与检查点无关的代码意味着，如果您的代码适用于一个检查点，那么它也适用于另一个检查点——只要它是针对类似任务训练的，即使架构不同。
<Tip>
请记住，架构是指模型的框架，检查点是给定架构的权重。例如，[BERT](https://huggingface.co/bert-base-uncased) 是一种架构，而 `bert-base-uncased` 是一个检查点。模型是一个通用术语，可以指代架构或检查点。
</Tip>
在本教程中，您将学习：
* 加载预训练的分词器 (Tokenizer)。
* 加载预训练的图像处理器 (Image Processor)。
* 加载预训练的特征提取器。
* 加载预训练的处理器。
* 加载预训练模型。
## AutoTokenizer
几乎每个 NLP 任务都始于一个分词器 (Tokenizer)。分词器 (Tokenizer)将输入转换为模型可以处理的格式。
使用 [`AutoTokenizer.from_pretrained`] 加载一个分词器 (Tokenizer)：
```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

然后按下面所示对输入进行分词：
```py
>>> sequence = "In a hole in the ground there lived a hobbit."
>>> print(tokenizer(sequence))
{'input_ids': [101, 1999, 1037, 4920, 1999, 1996, 2598, 2045, 2973, 1037, 7570, 10322, 4183, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

## AutoImageProcessor
对于视觉任务，图像处理器 (Image Processor)将图像处理为正确的输入格式。
```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```


## AutoFeatureExtractor
对于音频任务，特征提取器将音频信号处理为正确的输入格式。
使用 [`AutoFeatureExtractor.from_pretrained`] 加载一个特征提取器：
```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained(
...     "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
... )
```

## AutoProcessor
多模态任务需要一个结合了两种类型的预处理工具的处理器。例如，[LayoutLMV2](model_doc/layoutlmv2) 模型需要一个处理图像的处理器和一个处理文本的分词器 (Tokenizer)；处理器将它们结合在一起。
使用 [`AutoProcessor.from_pretrained`] 加载一个处理器：
```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
```

## AutoModel
<frameworkcontent> 
<pt> 最后，`AutoModelFor` 类让您可以加载给定任务的预训练模型（请参阅 [此处](model_doc/auto) 获取可用任务的完整列表）。例如，使用 [`AutoModelForSequenceClassification.from_pretrained`] 加载一个序列分类模型：
```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

轻松地重用相同的检查点来加载不同任务的架构：
```py
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased")
```

<提示 警告=真>
对于 PyTorch 模型，`from_pretrained()` 方法使用 `torch.load()`，它内部使用 `pickle`，已知存在安全问题。一般情况下，不要加载可能来自不受信任的源或可能被篡改的模型。这种安全风险对于托管在 Hugging Face Hub 上的公共模型部分得到了缓解，每次提交时都会对其进行 [恶意软件扫描](https://huggingface.co/docs/hub/security-malware)。请参阅 [Hub 文档](https://huggingface.co/docs/hub/security) 获取使用 GPG 的 [签名提交验证](https://huggingface.co/docs/hub/security-gpg#signing-commits-with-gpg) 等最佳实践。
TensorFlow 和 Flax 检查点不受影响，并且可以使用 `from_pretrained` 方法的 `from_tf` 和 `from_flax` 参数在 PyTorch 架构中加载它们以绕过此问题。
</Tip>
一般建议使用 `AutoTokenizer` 类和 `AutoModelFor` 类加载预训练模型的实例。这将确保您每次都加载正确的架构。在下一个 [教程](preprocessing) 中，了解如何使用新加载的分词器 (Tokenizer)、图像处理器 (Image Processor)、特征提取器和处理器对数据集进行预处理以进行微调。</pt> 
<tf> 

最后，`TFAutoModelFor` 类让您可以加载给定任务的预训练模型（请参阅 [此处](model_doc/auto) 获取可用任务的完整列表）。例如，使用 [`TFAutoModelForSequenceClassification.from_pretrained`] 加载一个序列分类模型：
```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

轻松地重用相同的检查点来加载不同任务的架构：
```py
>>> from transformers import TFAutoModelForTokenClassification

>>> model = TFAutoModelForTokenClassification.from_pretrained("distilbert-base-uncased")
```

一般建议使用 `AutoTokenizer` 类和 `TFAutoModelFor` 类加载预训练模型的实例。这将确保您每次都加载正确的架构。在下一个 [教程](preprocessing) 中，了解如何使用新加载的分词器 (Tokenizer)、图像处理器 (Image Processor)、特征提取器和处理器对数据集进行预处理以进行微调。</tf> </frameworkcontent>