<!--版权所有 2021 年 NVIDIA 公司和 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证发布的软件是按照“按原样”分发的，不附带任何明示或暗示的保证或条件。请参阅许可证了解特定语言下的权限和限制。
⚠️请注意，此文件是 Markdown 格式的，但包含特定的语法，用于我们的文档构建器（类似于 MDX），在您的 Markdown 查看器中可能无法正确显示。
-->
# QDQBERT

## 概述

QDQBERT 模型可以参考 Hao Wu、Patrick Judd、Xiaojie Zhang、Mikhail Isaev 和 PauliusMicikevicius 的《整数量化用于深度学习推断：原理和实证评估》（https://arxiv.org/abs/2004.09602）。

文章的摘要如下：
*量化技术可以减小深度神经网络的大小，通过利用高吞吐量的整数指令提高推断延迟和吞吐量。本文回顾了量化参数的数学方面，并评估了不同应用领域的各种神经网络模型的选择，包括视觉、语音和语言任务。我们专注于适用于高吞吐量整数运算管道的量化技术。我们还提出了一种 8 位量化的工作流程，该工作流程能够在所有研究的网络上保持与浮点基准的精度差异在 1%以内，包括更难量化的模型，如 MobileNets 和 BERT-large。* 

- QDQBERT 模型在 BERT 模型中向（i）线性层输入和权重，（ii）矩阵相乘输入，（iii）残差相加输入中添加了伪量化操作（一对 QuantizeLinear/DequantizeLinear 操作）。

- QDQBERT 需要依赖 [Pytorch Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization)。

要安装，请执行 `pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com` 


- QDQBERT 模型可以从 HuggingFace BERT 模型的任何检查点（例如 *bert-base-uncased*）加载，并执行量化感知训练/后训练量化。


- 使用 QDQBERT 模型对 SQUAD 任务进行量化感知训练和后训练量化的完整示例可在 [transformers/examples/research_projects/quantization-qdqbert/](examples/research_projects/quantization-qdqbert/) 中找到。



此模型由 [shangz](https://huggingface.co/shangz) 贡献。

### 设置默认量化器

QDQBERT 模型通过 [Pytorch Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization) 中的 `TensorQuantizer` 来添加伪量化操作（一对 QuantizeLinear/DequantizeLinear 操作）到 BERT 模型中。`TensorQuantizer` 是用于量化张量的模块，`QuantDescriptor` 定义了张量的量化方式。

有关详细信息，请参阅 [PytorchQuantization Toolkit 用户指南](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html)。

在创建 QDQBERT 模型之前，必须设置默认的 `QuantDescriptor`，定义默认的张量量化器。

示例：

```python
>>> import pytorch_quantization.nn as quant_nn
>>> from pytorch_quantization.tensor_quant import QuantDescriptor

>>> # The default tensor quantizer is set to use Max calibration method
>>> input_desc = QuantDescriptor(num_bits=8, calib_method="max")
>>> # The default tensor quantizer is set to be per-channel quantization for weights
>>> weight_desc = QuantDescriptor(num_bits=8, axis=((0,)))
>>> quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
>>> quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)
```

### 校准

校准是将数据样本传递给量化器，并确定张量的最佳缩放因子的术语。在设置张量量化器后，可以使用以下示例对模型进行校准：

```python
>>> # Find the TensorQuantizer and enable calibration
>>> for name, module in model.named_modules():
...     if name.endswith("_input_quantizer"):
...         module.enable_calib()
...         module.disable_quant()  # Use full precision data to calibrate

>>> # Feeding data samples
>>> model(x)
>>> # ...

>>> # Finalize calibration
>>> for name, module in model.named_modules():
...     if name.endswith("_input_quantizer"):
...         module.load_calib_amax()
...         module.enable_quant()

>>> # If running on GPU, it needs to call .cuda() again because new tensors will be created by calibration process
>>> model.cuda()

>>> # Keep running the quantized model
>>> # ...
```

### 导出到 ONNX

导出到 ONNX 的目标是通过 [TensorRT](https://developer.nvidia.com/tensorrt) 部署推断。伪量化将被拆分为一对 QuantizeLinear/DequantizeLinear 的 ONNX 操作。在将 TensorQuantizer 的静态成员设置为使用 Pytorch 自己的伪量化函数后，可以将伪量化的模型导出到 ONNX，按照 [torch.onnx](https://pytorch.org/docs/stable/onnx.html) 中的说明操作。示例：the instructions in [torch.onnx](https://pytorch.org/docs/stable/onnx.html). Example:

```python
>>> from pytorch_quantization.nn import TensorQuantizer

>>> TensorQuantizer.use_fb_fake_quant = True

>>> # Load the calibrated model
>>> ...
>>> # ONNX export
>>> torch.onnx.export(...)
```

## 文档资源
- [文本分类任务指南](../tasks/sequence_classification)
- [标记分类任务指南](../tasks/token_classification)
- [问答任务指南](../tasks/question_answering)
- [因果语言建模任务指南](../tasks/language_modeling)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)
- [多项选择任务指南](../tasks/multiple_choice)
## QDQBertConfig

[[autodoc]] QDQBertConfig

## QDQBertModel

[[autodoc]] QDQBertModel
    - forward

## QDQBertLMHeadModel

[[autodoc]] QDQBertLMHeadModel
    - forward

## QDQBertForMaskedLM

[[autodoc]] QDQBertForMaskedLM
    - forward

## QDQBertForSequenceClassification

[[autodoc]] QDQBertForSequenceClassification
    - forward

## QDQBertForNextSentencePrediction

[[autodoc]] QDQBertForNextSentencePrediction
    - forward

## QDQBertForMultipleChoice

[[autodoc]] QDQBertForMultipleChoice
    - forward

## QDQBertForTokenClassification

[[autodoc]] QDQBertForTokenClassification
    - forward

## QDQBertForQuestionAnswering

[[autodoc]] QDQBertForQuestionAnswering
    - forward
