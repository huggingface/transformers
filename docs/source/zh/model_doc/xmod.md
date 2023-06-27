<!--版权 2023 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非符合许可证要求，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”基础分发的，无论是明示的还是暗示的。请查看许可证以了解特定语言下的权限和限制。
⚠️请注意，此文件是 Markdown 格式的，但包含了我们的文档构建器的特定语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确显示。
-->
# X-MOD

## 概览

X-MOD 模型是由 Jonas Pfeiffer、Naman Goyal、Xi Lin、Xian Li、James Cross、Sebastian Riedel 和 Mikel Artetxe 在 [Lifting the Curse of Multilinguality by Pre-training Modular Transformers](http://dx.doi.org/10.18653/v1/2022.naacl-main.255) 中提出的。X-MOD 扩展了多语言遮蔽语言模型（如 [XLM-R](xlm-roberta)），在预训练期间包括了语言特定的模块化组件（_语言适配器_）。

在微调期间，每个变换器层中的语言适配器都是冻结的。

来自该论文的摘要如下:

*众所周知，多语言预训练模型在涵盖更多语言时会遭受多语言诅咒，导致每种语言的性能下降。我们通过引入语言特定模块来解决这个问题，这使我们能够增加模型的总容量，同时保持每种语言的可训练参数总数恒定。与先前的后续学习语言特定组件的工作相比，我们从一开始就对 Cross-lingual Modular（X-MOD）模型的模块进行预训练。我们在自然语言推理、命名实体识别和问答方面的实验表明，我们的方法不仅减轻了语言之间的负面干扰，还实现了积极的迁移，从而提高了单语和跨语言性能。此外，我们的方法能够在不降低性能的情况下后续添加语言，不再将模型使用限制为预训练语言集合。*

提示:- X-MOD 类似于 [XLM-R](xlm-roberta)，但不同之处在于需要指定输入语言以激活正确的语言适配器。- 主要模型——基础模型和大型模型——具有 81 种语言的适配器。

此模型由 [jvamvas](https://huggingface.co/jvamvas) 贡献。

原始代码可在 [此处](https://github.com/facebookresearch/fairseq/tree/58cc6cca18f15e6d56e3f60c959fe4f878960a60/fairseq/models/xmod) 找到，原始文档可在 [此处](https://github.com/facebookresearch/fairseq/tree/58cc6cca18f15e6d56e3f60c959fe4f878960a60/examples/xmod) 找到。

## 适配器使用

### 输入语言

有两种指定输入语言的方法: 1. 在使用模型之前设置默认语言:

```python
from transformers import XmodModel

model = XmodModel.from_pretrained("facebook/xmod-base")
model.set_default_language("en_XX")
```

2. 通过显式传递每个样本的语言适配器索引:
```python
import torch

input_ids = torch.tensor(
    [
        [0, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2],
        [0, 1310, 49083, 443, 269, 71, 5486, 165, 60429, 660, 23, 2],
    ]
)
lang_ids = torch.LongTensor(
    [
        0,  # en_XX
        8,  # de_DE
    ]
)
output = model(input_ids, lang_ids=lang_ids)
```

### 微调

论文建议在微调期间冻结嵌入层和语言适配器。提供了一种实现此目的的方法:
```python
model.freeze_embeddings_and_language_adapters()
# Fine-tune the model ...
```

### 跨语言迁移

在微调后，可以通过激活目标语言的语言适配器来测试零-shot 跨语言迁移:

```python
model.set_default_language("de_DE")
# Evaluate the model on German examples ...
```

## 资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)- [因果语言建模任务指南](../tasks/language_modeling)
- [遮蔽语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)

## XmodConfig

[[autodoc]] XmodConfig

## XmodModel

[[autodoc]] XmodModel
    - forward

## XmodForCausalLM

[[autodoc]] XmodForCausalLM
    - forward

## XmodForMaskedLM

[[autodoc]] XmodForMaskedLM
    - forward

## XmodForSequenceClassification

[[autodoc]] XmodForSequenceClassification
    - forward

## XmodForMultipleChoice

[[autodoc]] XmodForMultipleChoice
    - forward

## XmodForTokenClassification

[[autodoc]] XmodForTokenClassification
    - forward

## XmodForQuestionAnswering

[[autodoc]] XmodForQuestionAnswering
    - forward
