<!--Copyright 2021 NVIDIA Corporation and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# QDQBERT

<Tip warning={true}>

This model is in maintenance mode only, we don't accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

</Tip>

## Overview

The QDQBERT model can be referenced in [Integer Quantization for Deep Learning Inference: Principles and Empirical
Evaluation](https://arxiv.org/abs/2004.09602) by Hao Wu, Patrick Judd, Xiaojie Zhang, Mikhail Isaev and Paulius
Micikevicius.

The abstract from the paper is the following:

*Quantization techniques can reduce the size of Deep Neural Networks and improve inference latency and throughput by
taking advantage of high throughput integer instructions. In this paper we review the mathematical aspects of
quantization parameters and evaluate their choices on a wide range of neural network models for different application
domains, including vision, speech, and language. We focus on quantization techniques that are amenable to acceleration
by processors with high-throughput integer math pipelines. We also present a workflow for 8-bit quantization that is
able to maintain accuracy within 1% of the floating-point baseline on all networks studied, including models that are
more difficult to quantize, such as MobileNets and BERT-large.*

This model was contributed by [shangz](https://huggingface.co/shangz).

## Usage tips

- QDQBERT model adds fake quantization operations (pair of QuantizeLinear/DequantizeLinear ops) to (i) linear layer
  inputs and weights, (ii) matmul inputs, (iii) residual add inputs, in BERT model.
- QDQBERT requires the dependency of [Pytorch Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization). To install `pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com`
- QDQBERT model can be loaded from any checkpoint of HuggingFace BERT model (for example *google-bert/bert-base-uncased*), and
  perform Quantization Aware Training/Post Training Quantization.
- A complete example of using QDQBERT model to perform Quatization Aware Training and Post Training Quantization for
  SQUAD task can be found at [transformers/examples/research_projects/quantization-qdqbert/](examples/research_projects/quantization-qdqbert/).

### Set default quantizers

QDQBERT model adds fake quantization operations (pair of QuantizeLinear/DequantizeLinear ops) to BERT by
`TensorQuantizer` in [Pytorch Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization). `TensorQuantizer` is the module
for quantizing tensors, with `QuantDescriptor` defining how the tensor should be quantized. Refer to [Pytorch
Quantization Toolkit userguide](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html) for more details.

Before creating QDQBERT model, one has to set the default `QuantDescriptor` defining default tensor quantizers.

Example:

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

### Calibration

Calibration is the terminology of passing data samples to the quantizer and deciding the best scaling factors for
tensors. After setting up the tensor quantizers, one can use the following example to calibrate the model:

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

### Export to ONNX

The goal of exporting to ONNX is to deploy inference by [TensorRT](https://developer.nvidia.com/tensorrt). Fake
quantization will be broken into a pair of QuantizeLinear/DequantizeLinear ONNX ops. After setting static member of
TensorQuantizer to use Pytorch’s own fake quantization functions, fake quantized model can be exported to ONNX, follow
the instructions in [torch.onnx](https://pytorch.org/docs/stable/onnx.html). Example:

```python
>>> from pytorch_quantization.nn import TensorQuantizer

>>> TensorQuantizer.use_fb_fake_quant = True

>>> # Load the calibrated model
>>> ...
>>> # ONNX export
>>> torch.onnx.export(...)
```

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

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
