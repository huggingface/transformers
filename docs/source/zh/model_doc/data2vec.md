<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证，版本 2.0（“许可证”）授权；除非符合许可证的规定，否则不得使用此文件。您可以在以下位置获取许可证副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的担保或条件。请参阅许可证以获取特定语言的权限和限制。-->
⚠️ 注意，此文件是 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。
-->
# Data2Vec

## 概述

Data2Vec 模型是由 Alexei Baevski、Wei-Ning Hsu、Qiantong Xu、Arun Babu、Jiatao Gu 和 Michael Auli 在 [data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language](https://arxiv.org/pdf/2202.03555) 提出的。

Data2Vec 提出了一个统一的自监督学习框架，可以应用于不同的数据模态，包括文本、音频和图像。重要的是，预训练的预测目标是输入的上下文化潜在表示，而不是特定于模态的、上下文无关的目标。

论文中的摘要如下所示：

*虽然自监督学习的一般思想在不同的模态之间是相同的，但实际的算法和目标因为是为单一模态开发的而有很大的不同。为了让我们更接近通用的自监督学习，我们提出了 data2vec，这是一个框架，使用相同的学习方法来处理语音、NLP 或计算机视觉。其核心思想是基于输入的掩码视图来预测完整输入数据的潜在表示，以标准 Transformer 架构在自蒸馏设置下进行。data2vec 不预测诸如单词、视觉标记或人类语音单元之类的特定于模态的目标，它预测的是包含整个输入信息的上下文化潜在表示。在语音识别、图像分类和自然语言理解的主要基准测试中进行的实验表明，与主流方法相比，data2vec 取得了新的最先进或具有竞争力的性能。模型和代码可在 www.github.com/pytorch/fairseq/tree/master/examples/data2vec 获取。*

提示：

- Data2VecAudio、Data2VecText 和 Data2VecVision 都是使用相同的自监督学习方法进行训练的。- 对于 Data2VecAudio，预处理与 [`Wav2Vec2Model`] 相同，包括特征提取。- 对于 Data2VecText，预处理与 [`RobertaModel`] 相同，包括标记化。- 对于 Data2VecVision，预处理与 [`BeitModel`] 相同，包括特征提取。

此模型由 [edugp](https://huggingface.co/edugp) 和 [patrickvonplaten](https://huggingface.co/patrickvonplaten) 贡献。[sayakpaul](https://github.com/sayakpaul) 和 [Rocketknight1](https://github.com/Rocketknight1) 在 TensorFlow 中贡献了用于视觉的 Data2Vec。

原始代码（用于 NLP 和语音）可在 [此处](https://github.com/pytorch/fairseq/tree/main/examples/data2vec) 找到。视觉的原始代码可在 [此处](https://github.com/facebookresearch/data2vec_vision/tree/main/beit) 找到。

## 资源

以下是官方 Hugging Face 和社区（由 🌎 表示）资源列表，可帮助您开始使用 Data2Vec。
<PipelineTag pipeline="image-classification"/>
- [`Data2VecVisionForImageClassification`] 可使用此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) 和 [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) 进行支持。
- 要在自定义数据集上微调 [`TFData2VecVisionForImageClassification`]，请参阅 [此 notebook](https://colab.research.google.com/github/sayakpaul/TF-2.0-Hacks/blob/master/data2vec_vision_image_classification.ipynb)。
**Data2VecText 文档资源**
- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)- [因果语言建模任务指南](../tasks/language_modeling)- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)
**Data2VecAudio 文档资源**
- [音频分类任务指南](../tasks/audio_classification)
- [自动语音识别任务指南](../tasks/asr)
**Data2VecVision 文档资源**
- [图像分类](../tasks/image_classification)
- [语义分割](../tasks/semantic_segmentation)

如果您有兴趣提交资源以包含在此处，请随时提交拉取请求，我们将对其进行审查！资源理想情况下应展示出一些新的东西，而不是重复现有资源。
## Data2VecTextConfig

[[autodoc]] Data2VecTextConfig

## Data2VecAudioConfig

[[autodoc]] Data2VecAudioConfig

## Data2VecVisionConfig

[[autodoc]] Data2VecVisionConfig


## Data2VecAudioModel

[[autodoc]] Data2VecAudioModel
    - forward

## Data2VecAudioForAudioFrameClassification

[[autodoc]] Data2VecAudioForAudioFrameClassification
    - forward

## Data2VecAudioForCTC

[[autodoc]] Data2VecAudioForCTC
    - forward

## Data2VecAudioForSequenceClassification

[[autodoc]] Data2VecAudioForSequenceClassification
    - forward

## Data2VecAudioForXVector

[[autodoc]] Data2VecAudioForXVector
    - forward

## Data2VecTextModel

[[autodoc]] Data2VecTextModel
    - forward

## Data2VecTextForCausalLM

[[autodoc]] Data2VecTextForCausalLM
    - forward

## Data2VecTextForMaskedLM

[[autodoc]] Data2VecTextForMaskedLM
    - forward

## Data2VecTextForSequenceClassification

[[autodoc]] Data2VecTextForSequenceClassification
    - forward

## Data2VecTextForMultipleChoice

[[autodoc]] Data2VecTextForMultipleChoice
    - forward

## Data2VecTextForTokenClassification

[[autodoc]] Data2VecTextForTokenClassification
    - forward

## Data2VecTextForQuestionAnswering

[[autodoc]] Data2VecTextForQuestionAnswering
    - forward

## Data2VecVisionModel

[[autodoc]] Data2VecVisionModel
    - forward

## Data2VecVisionForImageClassification

[[autodoc]] Data2VecVisionForImageClassification
    - forward

## Data2VecVisionForSemanticSegmentation

[[autodoc]] Data2VecVisionForSemanticSegmentation
    - forward

## TFData2VecVisionModel

[[autodoc]] TFData2VecVisionModel
    - call

## TFData2VecVisionForImageClassification

[[autodoc]] TFData2VecVisionForImageClassification
    - call

## TFData2VecVisionForSemanticSegmentation

[[autodoc]] TFData2VecVisionForSemanticSegmentation
    - call
