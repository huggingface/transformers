<!--版权所有 2023 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2 版（“许可证”）获得许可；您除非遵守许可证，否则不得使用此文件。您可以在下面获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证发布的软件都是基于“按原样”提供的，不附带任何明示或暗示的担保或条件。有关许可证的详细信息，请参阅特定语言的权限和限制。
⚠️ 请注意，该文件是使用 Markdown 编写的，但包含了我们 doc-builder 的特定语法（类似于 MDX），可能在您的 Markdown 查看器中无法正确显示。
-->
# Pix2Struct

## 概述

Pix2Struct 模型是由 Kenton Lee、Mandar Joshi、Iulia Turc、Hexiang Hu、Fangyu Liu、Julian Eisenschlos、Urvashi Khandelwal、Peter Shaw、Ming-Wei Chang 和 Kristina Toutanova 在 [Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://arxiv.org/abs/2210.03347) 中提出的。

摘要如下：

> 视觉语境语言无处不在，来源包括带有图表的教科书、带有图像和表格的网页，以及带有按钮和表单的移动应用程序。也许由于这种多样性，以往的工作通常依赖于具有有限数据共享的特定领域方案、模型架构和目标。我们提出了 Pix2Struct，这是一个用于纯粹的视觉语境语言理解的预训练图像到文本模型，可以在包含视觉语境语言的任务中进行微调。Pix2Struct 的预训练目标是将屏幕截图中的蒙版解析为简化的 HTML。作为预训练数据源，Web 具有丰富的视觉元素，这些元素在 HTML 结构中清晰地反映出来，非常适合于各种下游任务的多样性。直观地说，这个目标包含了常见的预训练信号，如 OCR、语言建模和图像字幕。除了新颖的预训练策略，我们还引入了可变分辨率的输入表示和更灵活的语言和视觉输入集成，其中语言提示（如问题）直接呈现在输入图像的上方。我们首次展示了单个预训练模型在四个领域的九个任务中有六个任务达到了最先进的结果，这四个领域分别是文档、插图、用户界面和自然图像。

提示：

Pix2Struct 已在各种任务和数据集上进行了微调，包括图像字幕、视觉问答（VQA）以及不同输入类型（书籍、图表、科学图解），UI 组件标题等。完整列表可参见论文中的表格 1。因此，我们建议您将这些模型用于它们已经进行微调的任务。例如，如果您想将 Pix2Struct 用于 UI 标题，应使用在 UI 数据集上进行微调的模型。如果您想将 Pix2Struct 用于图像字幕，应使用在自然图像字幕数据集上进行微调的模型，依此类推。

如果您想使用模型进行条件文本标题，请确保使用 `add_special_tokens=False` 的处理器。
此模型由 [ybelkada](https://huggingface.co/ybelkada) 贡献。原始代码可在 [此处](https://github.com/google-research/pix2struct) 找到。

## 资源

- [微调笔记本](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_pix2struct.ipynb)
- [所有模型](https://huggingface.co/models?search=pix2struct)

## Pix2StructConfig

[[autodoc]] Pix2StructConfig
    - from_text_vision_configs

## Pix2StructTextConfig

[[autodoc]] Pix2StructTextConfig

## Pix2StructVisionConfig

[[autodoc]] Pix2StructVisionConfig

## Pix2StructProcessor

[[autodoc]] Pix2StructProcessor

## Pix2StructImageProcessor

[[autodoc]] Pix2StructImageProcessor
    - preprocess

## Pix2StructTextModel

[[autodoc]] Pix2StructTextModel
    - forward

## Pix2StructVisionModel

[[autodoc]] Pix2StructVisionModel
    - forward

## Pix2StructForConditionalGeneration

[[autodoc]] Pix2StructForConditionalGeneration
    - forward
