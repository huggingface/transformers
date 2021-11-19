.. 
    Copyright 2021 NVIDIA Corporation and The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

QDQBERT
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The QDQBERT model can be referenced in `Integer Quantization for Deep Learning Inference: Principles and Empirical
Evaluation <https://arxiv.org/abs/2004.09602>`__ by Hao Wu, Patrick Judd, Xiaojie Zhang, Mikhail Isaev and Paulius
Micikevicius.

The abstract from the paper is the following:

*Quantization techniques can reduce the size of Deep Neural Networks and improve inference latency and throughput by
taking advantage of high throughput integer instructions. In this paper we review the mathematical aspects of
quantization parameters and evaluate their choices on a wide range of neural network models for different application
domains, including vision, speech, and language. We focus on quantization techniques that are amenable to acceleration
by processors with high-throughput integer math pipelines. We also present a workflow for 8-bit quantization that is
able to maintain accuracy within 1% of the floating-point baseline on all networks studied, including models that are
more difficult to quantize, such as MobileNets and BERT-large.*

Tips:

- QDQBERT model adds fake quantization operations (pair of QuantizeLinear/DequantizeLinear ops) to (i) linear layer
  inputs and weights, (ii) matmul inputs, (iii) residual add inputs, in BERT model.

- QDQBERT requires the dependency of `Pytorch Quantization Toolkit
  <https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization>`__. To install ``pip install
  pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com``

- QDQBERT model can be loaded from any checkpoint of HuggingFace BERT model (for example *bert-base-uncased*), and
  perform Quantization Aware Training/Post Training Quantization.

- A complete example of using QDQBERT model to perform Quatization Aware Training and Post Training Quantization for
  SQUAD task can be found at `transformers/examples/research_projects/quantization-qdqbert/
  </examples/research_projects/quantization-qdqbert/>`_.

This model was contributed by `shangz <https://huggingface.co/shangz>`__.


Set default quantizers
_______________________________________________________________________________________________________________________

QDQBERT model adds fake quantization operations (pair of QuantizeLinear/DequantizeLinear ops) to BERT by
:obj:`TensorQuantizer` in `Pytorch Quantization Toolkit
<https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization>`__. :obj:`TensorQuantizer` is the module
for quantizing tensors, with :obj:`QuantDescriptor` defining how the tensor should be quantized. Refer to `Pytorch
Quantization Toolkit userguide
<https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html>`__ for more details.

Before creating QDQBERT model, one has to set the default :obj:`QuantDescriptor` defining default tensor quantizers.
Example:

.. code-block::

    >>> import pytorch_quantization.nn as quant_nn
    >>> from pytorch_quantization.tensor_quant import QuantDescriptor

    >>> # The default tensor quantizer is set to use Max calibration method
    >>> input_desc = QuantDescriptor(num_bits=8, calib_method="max")
    >>> # The default tensor quantizer is set to be per-channel quantization for weights
    >>> weight_desc = QuantDescriptor(num_bits=8, axis=((0,)))
    >>> quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
    >>> quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)


Calibration
_______________________________________________________________________________________________________________________

Calibration is the terminology of passing data samples to the quantizer and deciding the best scaling factors for
tensors. After setting up the tensor quantizers, one can use the following example to calibrate the model:

.. code-block::

    >>> # Find the TensorQuantizer and enable calibration
    >>> for name, module in model.named_modules():
    >>>     if name.endswith('_input_quantizer'):
    >>>         module.enable_calib()
    >>>         module.disable_quant()  # Use full precision data to calibrate

    >>> # Feeding data samples
    >>> model(x)
    >>> # ...

    >>> # Finalize calibration
    >>> for name, module in model.named_modules():
    >>>     if name.endswith('_input_quantizer'):
    >>>         module.load_calib_amax()
    >>>         module.enable_quant()

    >>> # If running on GPU, it needs to call .cuda() again because new tensors will be created by calibration process
    >>> model.cuda()

    >>> # Keep running the quantized model
    >>> # ...


Export to ONNX
_______________________________________________________________________________________________________________________

The goal of exporting to ONNX is to deploy inference by `TensorRT <https://developer.nvidia.com/tensorrt>`__. Fake
quantization will be broken into a pair of QuantizeLinear/DequantizeLinear ONNX ops. After setting static member of
TensorQuantizer to use Pytorch’s own fake quantization functions, fake quantized model can be exported to ONNX, follow
the instructions in `torch.onnx <https://pytorch.org/docs/stable/onnx.html>`__. Example:

.. code-block::

    >>> from pytorch_quantization.nn import TensorQuantizer
    >>> TensorQuantizer.use_fb_fake_quant = True

    >>> # Load the calibrated model
    >>> ...
    >>> # ONNX export
    >>> torch.onnx.export(...)


QDQBertConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.QDQBertConfig
    :members:


QDQBertModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.QDQBertModel
    :members: forward


QDQBertLMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.QDQBertLMHeadModel
    :members: forward


QDQBertForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.QDQBertForMaskedLM
    :members: forward


QDQBertForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.QDQBertForSequenceClassification
    :members: forward


QDQBertForNextSentencePrediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.QDQBertForNextSentencePrediction
    :members: forward


QDQBertForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.QDQBertForMultipleChoice
    :members: forward


QDQBertForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.QDQBertForTokenClassification
    :members: forward


QDQBertForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.QDQBertForQuestionAnswering
    :members: forward

