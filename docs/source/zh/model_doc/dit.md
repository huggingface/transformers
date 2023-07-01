<!--版权所有2022年HuggingFace团队保留所有权利。-->
根据Apache许可证2.0版（“许可证”）许可，除非符合许可证的规定，否则不得使用此文件。您可以在以下位置获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的保证或条件。请参阅许可证以获取特定语言下的权限和限制。⚠️请注意，此文件是Markdown格式，但包含我们的文档生成器（类似于MDX）的特定语法，可能无法在Markdown查看器中正确呈现。
-->

# DiT

## 概述

DiT由Junlong Li，Yiheng Xu，Tengchao Lv，Lei Cui，Cha Zhang和Furu Wei在[DiT:自监督预训练文档图像Transformer](https://arxiv.org/abs/2203.02378)中提出。

DiT将[BEiT](beit)（图像Transformer的BERT预训练）的自监督目标应用于4200万个文档图像，从而在以下任务中实现了最先进的结果：

- 文档图像分类：[RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/)数据集（包含400,000张属于16个类别的图像）。
- 文档布局分析：[PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)数据集（由自动解析PubMed XML文件构建的超过360,000个文档图像）。
- 表格检测：[ICDAR 2019 cTDaR](https://github.com/cndplab-founder/ICDAR2019_cTDaR)数据集（包含600个训练图像和240个测试图像）。

论文中的摘要如下所示：
*Image Transformer最近在自然图像理解方面取得了重大进展，无论是使用受监督（ViT、DeiT等）还是自监督（BEiT、MAE等）的预训练技术。在本文中，我们提出了DiT，这是一种使用大规模未标记文本图像进行文档AI任务的自监督预训练文档图像Transformer模型，这对于由于缺乏人工标记的文档图像而不存在受监督的对应模型是至关重要的。我们将DiT作为骨干网络用于各种基于视觉的文档AI任务，包括文档图像分类、文档布局分析以及表格检测。实验结果表明，经过自监督预训练的DiT模型在这些下游任务上获得了新的最先进结果，例如文档图像分类（91.11 → 92.69）、文档布局分析（91.0 → 94.9）和表格检测（94.23 → 96.55）。*
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/dit_architecture.jpg"
alt="drawing" width="600"/> 

<small>方法概述。摘自[原始论文](https://arxiv.org/abs/2203.02378)。</small>

可以直接使用DiT的权重进行AutoModel API：alt="drawing" width="600"/> 

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("microsoft/dit-base")
```

这将加载在掩码图像建模上预训练的模型。请注意，这不包括在顶部用于预测视觉令牌的语言建模头。
要包含头部，可以将权重加载到`BeitForMaskedImageModeling`模型中，如下所示：

```python
from transformers import BeitForMaskedImageModeling

model = BeitForMaskedImageModeling.from_pretrained("microsoft/dit-base")
```

您还可以从[hub](https://huggingface.co/models?other=dit)加载经过微调的模型，如下所示：

```python
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
```
此特定检查点是在[RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/)上进行微调的，这是一个重要的文档图像分类基准。
关于文档图像分类的推理示例可以在[这里](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DiT/Inference_with_DiT_(Document_Image_Transformer)_for_document_image_classification.ipynb)找到。

由于DiT的架构与BEiT相同，因此可以参考[BEiT的文档页面](beit)获取所有提示、代码示例和笔记本。

此模型由[nielsr](https://huggingface.co/nielsr)贡献。原始代码可以在[这里](https://github.com/microsoft/unilm/tree/master/dit)找到。





## 资源

以下是官方Hugging Face和社区（由🌎表示）资源列表，可帮助您开始使用DiT。
<PipelineTag pipeline="image-classification"/>
- [`BeitForImageClassification`](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)由此[示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)和[笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)支持。

如果您有兴趣提交要包含在此处的资源，请随时提出拉取请求，我们将对其进行审核！该资源应该展示一些新内容，而不是重复现有资源。

- [`BeitForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).

如果您有兴趣提交一份资源以包含在这里，请随时发起一个Pull Request，我们会进行审核！这份资源最好能展示一些新的内容，而不是重复现有的资源。