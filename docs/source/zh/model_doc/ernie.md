<!-- 版权所有2022年HuggingFace团队保留所有权利。
根据 Apache 许可证第2版（“许可证”）进行许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的保证或条件。有关许可证下的具体语言权限和限制，请参阅许可证。an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
请注意，此文件是Markdown格式，但包含了特定的语法用于我们的文档构建器（类似MDX），可能无法在您的Markdown查看器中正确呈现。
⚠️ 请注意，此文件是Markdown格式，但包含了特定的语法用于我们的文档构建器（类似MDX），可能无法在您的Markdown查看器中正确呈现。请注意，此文件是Markdown格式，但包含了特定的语法用于我们的文档构建器（类似MDX），可能无法在您的Markdown查看器中正确呈现。
-->
# ERNIE

## 概述

ERNIE是百度提出的一系列强大的模型，尤其在中文任务中表现出色。包括
[ERNIE1.0](https://arxiv.org/abs/1904.09223)，[ERNIE2.0](https://ojs.aaai.org/index.php/AAAI/article/view/6428)，
[ERNIE3.0](https://arxiv.org/abs/2107.02137)，[ERNIE-Gram](https://arxiv.org/abs/2010.12148)，
[ERNIE-health](https://arxiv.org/abs/2110.07244)等等。

这些模型由[nghuyong](https://huggingface.co/nghuyong)贡献，官方代码可以在[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)（在PaddlePaddle中）找到。

### 如何使用

以`ernie-1.0-base-zh`为例：
```Python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
model = AutoModel.from_pretrained("nghuyong/ernie-1.0-base-zh")
```

### 支持的模型

|     模型名称      | 语言 |           描述           ||:-------------------:|:--------:|:-------------------------------:||  ernie-1.0-base-zh  | 中文  | 层数:12，头数:12，隐藏层:768 ||  ernie-2.0-base-en  | 英文  | 层数:12，头数:12，隐藏层:768 || ernie-2.0-large-en  | 英文  | 层数:24，头数:16，隐藏层:1024 ||  ernie-3.0-base-zh  | 中文  | 层数:12，头数:12，隐藏层:768 || ernie-3.0-medium-zh | 中文  | 层数:6，头数:12，隐藏层:768 ||  ernie-3.0-mini-zh  | 中文  | 层数:6，头数:12，隐藏层:384 || ernie-3.0-micro-zh  | 中文  | 层数:4，头数:12，隐藏层:384 ||  ernie-3.0-nano-zh  | 中文  | 层数:4，头数:12，隐藏层:312 ||   ernie-health-zh   | 中文  | 层数:12，头数:12，隐藏层:768 ||    ernie-gram-zh    | 中文  | 层数:12，头数:12，隐藏层:768 |

您可以在Hugging Face的模型中心找到所有支持的模型：[huggingface.co/nghuyong](https://huggingface.co/nghuyong)，并从Paddle的官方仓库获取模型的详细信息：[PaddleNLP](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers/ERNIE/contents.html)和[ERNIE](https://github.com/PaddlePaddle/ERNIE/blob/repro)。

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)- [因果语言建模任务指南](../tasks/language_modeling)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多选任务指南](../tasks/multiple_choice)
## ErnieConfig

[[autodoc]] ErnieConfig
    - all

## Ernie specific outputs

[[autodoc]] models.ernie.modeling_ernie.ErnieForPreTrainingOutput

## ErnieModel

[[autodoc]] ErnieModel
    - forward

## ErnieForPreTraining

[[autodoc]] ErnieForPreTraining
    - forward

## ErnieForCausalLM

[[autodoc]] ErnieForCausalLM
    - forward

## ErnieForMaskedLM

[[autodoc]] ErnieForMaskedLM
    - forward

## ErnieForNextSentencePrediction

[[autodoc]] ErnieForNextSentencePrediction
    - forward

## ErnieForSequenceClassification

[[autodoc]] ErnieForSequenceClassification
    - forward

## ErnieForMultipleChoice

[[autodoc]] ErnieForMultipleChoice
    - forward

## ErnieForTokenClassification

[[autodoc]] ErnieForTokenClassification
    - forward

## ErnieForQuestionAnswering

[[autodoc]] ErnieForQuestionAnswering
    - forward