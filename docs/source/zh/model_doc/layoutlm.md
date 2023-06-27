<!--版权所有 2020 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；您除非遵守许可证，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，否则根据许可证分发的软件是按“按原样”分发，不附带任何形式的担保或条件。请参阅许可证以了解特定的语言许可权限和限制。
⚠️请注意，此文件是 Markdown 格式的，但包含我们的 doc-builder（类似于 MDX）的特定语法，您的 Markdown 查看器可能无法正确呈现它。
-->
# LayoutLM

<a id='Overview'></a>

## 概述（Overview）

LayoutLM 模型是由 Yiheng Xu，Minghao Li，Lei Cui，Shaohan Huang，Furu Wei 和 Ming Zhou 在“LayoutLM：预训练文本和布局用于文档图像理解”（https://arxiv.org/abs/1912.13318）中提出的。它是一种简单而有效的文档图像理解和信息提取任务（例如表单理解和收据理解）的预训练方法。它在几个下游任务上获得了最先进的结果：

- 表单理解：[FUNSD](https://guillaumejaume.github.io/FUNSD/) 数据集（包含 199 个标注
表单，涵盖 3 万多个单词）。

- 收据理解：[SROIE](https://rrc.cvc.uab.es/?ch=13) 数据集（包含 626 个训练  training and 347 receipts for testing).

- 文档图像分类：[RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) 数据集（包含  40 万张属于 16 个类别的图像）。

下面是该论文的摘要：

*近年来，在各种 NLP 任务中已经成功验证了预训练技术。尽管预训练模型在 NLP 应用中被广泛使用，但它们几乎完全集中在文本级别的操作上，而忽略了对文档图像理解至关重要的布局和样式信息。在本文中，我们提出 LayoutLM 模型，以联合建模扫描文档图像中的文本和布局信息之间的交互，这对于很多实际的文档图像理解任务（例如从扫描文档中提取信息）非常有益。此外，我们还利用图像特征将单词的视觉信息融入到 LayoutLM 中。据我们所知，这是首次在单一框架中联合学习文本和布局级别预训练。它在几个下游任务中取得了最新的最佳结果，包括表单理解（从 70.72 提高到 79.27）、收据理解（从 94.02 提高到 95.24）和文档图像分类（从 93.07 提高到 94.42）.*

提示：

- 除了 *input_ids* 之外，[`~transformers.LayoutLMModel.forward`] 还需要输入 `bbox`，  它是输入标记的边界框（即 2D 位置）。

可以使用外部 OCR 引擎（例如 Google 的 [Tesseract]  （https://github.com/tesseract-ocr/tesseract））获取这些边界框（有一个可用的  [Python 封装器](https://pypi.org/project/pytesseract/)）。

每个边界框应该采用（x0，y0，x1，y1）的格式，  其中（x0，y0）对应于边界框左上角的位置，（x1，y1）表示  右下角的位置。请注意，首先需要将边界框归一化为 0-1000 的比例。您可以使用以下函数进行归一化：

```python
def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]
```

这里，`width` 和 `height` 对应于标记出现的原始文档的宽度和高度。可以使用 Python Image Library（PIL）库获取它们，例如：
```python
from PIL import Image

# Document can be a png, jpg, etc. PDFs must be converted to images.
image = Image.open(name_of_your_document).convert("RGB")

width, height = image.size
```

## 资源

下面是官方 Hugging Face 和社区（用🌎表示）资源的列表，可帮助您开始使用 LayoutLM。如果您有兴趣提交要包含在此处的资源，请随时打开 Pull Request，我们将进行审核！资源应该展示出新的东西，而不是重复现有的资源。

<PipelineTag pipeline="document-question-answering" />

- 有关使用 Keras 和 Hugging Face Transformers 进行文档理解的 [微调  LayoutLM 的博客文章](https://www.philschmid.de/fine-tuning-layoutlm-keras)。  Transformers](https://www.philschmid.de/fine-tuning-layoutlm-keras).

- 有关如何 [仅使用 Hugging Face Transformers 微调 LayoutLM 进行文档理解的博客文章](https://www.philschmid.de/fine-tuning-layoutlm)。

- 如何 [使用图像嵌入在 FUNSD 数据集上微调 LayoutLM 的笔记本](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Add_image_embeddings_to_LayoutLM.ipynb)。

- 另请参阅：[文档问答任务指南](../tasks/document_question_answering)

<PipelineTag pipeline="text-classification" />

- 如何 [在 RVL-CDIP 数据集上对 LayoutLM 进行序列分类微调的笔记本](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Fine_tuning_LayoutLMForSequenceClassification_on_RVL_CDIP.ipynb)。
- [文本分类任务指南](../tasks/sequence_classification)
<PipelineTag pipeline="token-classification" />

- 如何 [在 FUNSD 数据集上对 LayoutLM 进行标记分类微调的笔记本](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Fine_tuning_LayoutLMForTokenClassification_on_FUNSD.ipynb)。

- [标记分类任务指南](../tasks/token_classification)

**其他资源**

- [掩码语言模型任务指南](../tasks/masked_language_modeling)

🚀 部署

- 如何 [使用 Hugging Face 推理端点部署 LayoutLM](https://www.philschmid.de/inference-endpoints-layoutlm) 的博客文章。

## LayoutLMConfig
[[autodoc]] LayoutLMConfig
## LayoutLMTokenizer
[[autodoc]] LayoutLMTokenizer
## LayoutLMTokenizerFast
[[autodoc]] LayoutLMTokenizerFast
## LayoutLMModel
[[autodoc]] LayoutLMModel
## LayoutLMForMaskedLM
[[autodoc]] LayoutLMForMaskedLM
## LayoutLMForSequenceClassification
[[autodoc]] LayoutLMForSequenceClassification
## LayoutLMForTokenClassification
[[autodoc]] LayoutLMForTokenClassification
## LayoutLMForQuestionAnswering
[[autodoc]] LayoutLMForQuestionAnswering
## TFLayoutLMModel
[[autodoc]] TFLayoutLMModel
## TFLayoutLMForMaskedLM
[[autodoc]] TFLayoutLMForMaskedLM
## TFLayoutLMForSequenceClassification
[[autodoc]] TFLayoutLMForSequenceClassification
## TFLayoutLMForTokenClassification
[[autodoc]] TFLayoutLMForTokenClassification
## TFLayoutLMForQuestionAnswering
[[autodoc]] TFLayoutLMForQuestionAnswering