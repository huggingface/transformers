<!--版权所有2021年HuggingFace团队。保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非符合许可证的规定，否则您不得使用此文件。您可以在以下网址获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“原样”分发，不附带任何明示或暗示的保证或条件。请参阅许可证以获取特定语言下的权限和限制。请注意，此文件是 Markdown 格式的，但包含我们 doc-builder 的特定语法（类似于 MDX），在您的 Markdown 查看器中可能无法
正确渲染。-->


# VisualBERT

## 概述

VisualBERT 模型是由 Liunian Harold Li、Mark Yatskar、Da Yin、Cho-Jui Hsieh 和 Kai-Wei Chang 在 [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/pdf/1908.03557) 一文中提出的。VisualBERT 是一个基于各种（图像，文本）对进行训练的神经网络。

以下是该论文的摘要：

*我们提出了 VisualBERT，这是一个用于建模广泛范围的视觉和语言任务的简单而灵活的框架。VisualBERT 由一堆 Transformer 层组成，它使用自注意力机制隐式地对输入文本和相关输入图像的元素进行对齐。我们还为在图像字幕数据上进行 VisualBERT 的预训练提出了两个基于视觉的语言模型目标。在包括 VQA、VCR、NLVR2 和 Flickr30K 在内的四个视觉和语言任务上的实验证明，VisualBERT 在性能上超过了或与最先进的模型相媲美，同时更为简单。进一步的分析表明，VisualBERT 可以将语言元素与图像区域相关联，而无需任何明确的监督，并且甚至对语法关系敏感，例如跟踪动词与对应其参数的图像区域之间的关联。* 

提示：

1. 提供的大多数检查点与 [`VisualBertForPreTraining`] 配置兼容。其他提供的检查点是用于下游任务（如 VQA（'visualbert-vqa'），VCR（'visualbert-vcr'），NLVR2（'visualbert-nlvr2'））的微调检查点。因此，如果您不是在进行这些下游任务，建议使用预训练的检查点。

2. 对于 VCR 任务，作者使用了经过微调的检测器来生成视觉嵌入，对于所有检查点都是如此。

我们没有将检测器及其权重作为软件包的一部分提供，但它将在研究项目中提供，并且可以直接加载到提供的检测器中。   

## 用法

VisualBERT 是一个多模态的视觉和语言模型。它可用于视觉问答、多项选择、视觉推理和区域到短语对齐等任务。VisualBERT 使用类似 BERT 的 transformer 为图像文本对准备嵌入。文本和视觉特征都被投影到相同的潜空间中。

要将图像馈送给模型，需要将每个图像通过预训练的对象检测器，并提取出区域和边界框。作者使用在通过这些区域经过预训练的 CNN（如 ResNet）之后生成的特征作为视觉嵌入。他们还添加了绝对位置嵌入，并将结果的向量序列与标准 BERT 模型一起馈送。文本输入以 BERT 中的 [CLS] 和 [SEP] 标记为边界，并期望被分段 ID 适当地设置为文本和视觉部分。

使用 [`BertTokenizer`] 对文本进行编码。必须使用自定义的检测器/图像处理器 (Image Processor)来获取视觉嵌入。以下示例笔记本展示了如何使用具有类似 Detectron 的模型的 VisualBERT：

- [VisualBERT VQA 演示笔记本](https://github.com/huggingface/transformers/tree/main/examples/research_projects/visual_bert)：此笔记本  包含了 VisualBERT VQA 的示例。

- [为 VisualBERT 生成嵌入（Colab 笔记本）](https://colab.research.google.com/drive/1bLGxKdldwqnMVA5x4neY7-l_8fKGWQYI?usp=sharing)：此笔记本  包含了如何生成视觉嵌入的示例。

以下示例显示如何使用 [`VisualBertModel`] 获取最后一个隐藏状态：

```python
>>> import torch
>>> from transformers import BertTokenizer, VisualBertModel

>>> model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
>>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

>>> inputs = tokenizer("What is the man eating?", return_tensors="pt")
>>> # this is a custom function that returns the visual embeddings given the image path
>>> visual_embeds = get_visual_embeddings(image_path)

>>> visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
>>> visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
>>> inputs.update(
...     {
...         "visual_embeds": visual_embeds,
...         "visual_token_type_ids": visual_token_type_ids,
...         "visual_attention_mask": visual_attention_mask,
...     }
... )
>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
```

此模型由 [gchhablani](https://huggingface.co/gchhablani) 提供。

原始代码可以在 [此处](https://github.com/uclanlp/visualbert) 找到。


## VisualBertConfig

[[autodoc]] VisualBertConfig

## VisualBertModel

[[autodoc]] VisualBertModel
    - forward

## VisualBertForPreTraining

[[autodoc]] VisualBertForPreTraining
    - forward

## VisualBertForQuestionAnswering

[[autodoc]] VisualBertForQuestionAnswering
    - forward

## VisualBertForMultipleChoice

[[autodoc]] VisualBertForMultipleChoice
    - forward

## VisualBertForVisualReasoning

[[autodoc]] VisualBertForVisualReasoning
    - forward

## VisualBertForRegionToPhraseAlignment

[[autodoc]] VisualBertForRegionToPhraseAlignment
    - forward