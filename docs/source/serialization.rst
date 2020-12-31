.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Exporting transformers models
***********************************************************************************************************************

ONNX / ONNXRuntime
=======================================================================================================================

Projects `ONNX (Open Neural Network eXchange) <http://onnx.ai>`_ and `ONNXRuntime (ORT)
<https://microsoft.github.io/onnxruntime/>`_ are part of an effort from leading industries in the AI field to provide a
unified and community-driven format to store and, by extension, efficiently execute neural network leveraging a variety
of hardware and dedicated optimizations.

Starting from transformers v2.10.0 we partnered with ONNX Runtime to provide an easy export of transformers models to
the ONNX format. You can have a look at the effort by looking at our joint blog post `Accelerate your NLP pipelines
using Hugging Face Transformers and ONNX Runtime
<https://medium.com/microsoftazure/accelerate-your-nlp-pipelines-using-hugging-face-transformers-and-onnx-runtime-2443578f4333>`_.

Exporting a model is done through the script `convert_graph_to_onnx.py` at the root of the transformers sources. The
following command shows how easy it is to export a BERT model from the library, simply run:

.. code-block:: bash

    python convert_graph_to_onnx.py --framework <pt, tf> --model bert-base-cased bert-base-cased.onnx

The conversion tool works for both PyTorch and Tensorflow models and ensures:

* The model and its weights are correctly initialized from the Hugging Face model hub or a local checkpoint.
* The inputs and outputs are correctly generated to their ONNX counterpart.
* The generated model can be correctly loaded through onnxruntime.

.. note::
    Currently, inputs and outputs are always exported with dynamic sequence axes preventing some optimizations on the
    ONNX Runtime. If you would like to see such support for fixed-length inputs/outputs, please open up an issue on
    transformers.


Also, the conversion tool supports different options which let you tune the behavior of the generated model:

* **Change the target opset version of the generated model.** (More recent opset generally supports more operators and
  enables faster inference)

* **Export pipeline-specific prediction heads.** (Allow to export model along with its task-specific prediction
  head(s))

* **Use the external data format (PyTorch only).** (Lets you export model which size is above 2Gb (`More info
  <https://github.com/pytorch/pytorch/pull/33062>`_))


Optimizations
-----------------------------------------------------------------------------------------------------------------------

ONNXRuntime includes some transformers-specific transformations to leverage optimized operations in the graph. Below
are some of the operators which can be enabled to speed up inference through ONNXRuntime (*see note below*):

* Constant folding
* Attention Layer fusing
* Skip connection LayerNormalization fusing
* FastGeLU approximation

Some of the optimizations performed by ONNX runtime can be hardware specific and thus lead to different performances if
used on another machine with a different hardware configuration than the one used for exporting the model. For this
reason, when using ``convert_graph_to_onnx.py`` optimizations are not enabled, ensuring the model can be easily
exported to various hardware. Optimizations can then be enabled when loading the model through ONNX runtime for
inference.


.. note::
    When quantization is enabled (see below), ``convert_graph_to_onnx.py`` script will enable optimizations on the
    model because quantization would modify the underlying graph making it impossible for ONNX runtime to do the
    optimizations afterwards.

.. note::
    For more information about the optimizations enabled by ONNXRuntime, please have a look at the `ONNXRuntime Github
    <https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers>`_.

Quantization
-----------------------------------------------------------------------------------------------------------------------

ONNX exporter supports generating a quantized version of the model to allow efficient inference.

Quantization works by converting the memory representation of the parameters in the neural network to a compact integer
format. By default, weights of a neural network are stored as single-precision float (`float32`) which can express a
wide-range of floating-point numbers with decent precision. These properties are especially interesting at training
where you want fine-grained representation.

On the other hand, after the training phase, it has been shown one can greatly reduce the range and the precision of
`float32` numbers without changing the performances of the neural network.

More technically, `float32` parameters are converted to a type requiring fewer bits to represent each number, thus
reducing the overall size of the model. Here, we are enabling `float32` mapping to `int8` values (a non-floating,
single byte, number representation) according to the following formula:

.. math::
    y_{float32} = scale * x_{int8} - zero\_point

.. note::
    The quantization process will infer the parameter `scale` and `zero_point` from the neural network parameters

Leveraging tiny-integers has numerous advantages when it comes to inference:

* Storing fewer bits instead of 32 bits for the `float32` reduces the size of the model and makes it load faster.
* Integer operations execute a magnitude faster on modern hardware
* Integer operations require less power to do the computations

In order to convert a transformers model to ONNX IR with quantized weights you just need to specify ``--quantize`` when
using ``convert_graph_to_onnx.py``. Also, you can have a look at the ``quantize()`` utility-method in this same script
file.

Example of quantized BERT model export:

.. code-block:: bash

    python convert_graph_to_onnx.py --framework <pt, tf> --model bert-base-cased --quantize bert-base-cased.onnx

.. note::
    Quantization support requires ONNX Runtime >= 1.4.0

.. note::
    When exporting quantized model you will end up with two different ONNX files. The one specified at the end of the
    above command will contain the original ONNX model storing `float32` weights. The second one, with ``-quantized``
    suffix, will hold the quantized parameters.


TorchScript
=======================================================================================================================

.. note::
    This is the very beginning of our experiments with TorchScript and we are still exploring its capabilities with
    variable-input-size models. It is a focus of interest to us and we will deepen our analysis in upcoming releases,
    with more code examples, a more flexible implementation, and benchmarks comparing python-based codes with compiled
    TorchScript.


According to Pytorch's documentation: "TorchScript is a way to create serializable and optimizable models from PyTorch
code". Pytorch's two modules `JIT and TRACE <https://pytorch.org/docs/stable/jit.html>`_ allow the developer to export
their model to be re-used in other programs, such as efficiency-oriented C++ programs.

We have provided an interface that allows the export of 🤗 Transformers models to TorchScript so that they can be reused
in a different environment than a Pytorch-based python program. Here we explain how to export and use our models using
TorchScript.

Exporting a model requires two things:

* a forward pass with dummy inputs.
* model instantiation with the ``torchscript`` flag.

These necessities imply several things developers should be careful about. These are detailed below.


Implications
-----------------------------------------------------------------------------------------------------------------------

TorchScript flag and tied weights
-----------------------------------------------------------------------------------------------------------------------

This flag is necessary because most of the language models in this repository have tied weights between their
``Embedding`` layer and their ``Decoding`` layer. TorchScript does not allow the export of models that have tied
weights, therefore it is necessary to untie and clone the weights beforehand.

This implies that models instantiated with the ``torchscript`` flag have their ``Embedding`` layer and ``Decoding``
layer separate, which means that they should not be trained down the line. Training would de-synchronize the two
layers, leading to unexpected results.

This is not the case for models that do not have a Language Model head, as those do not have tied weights. These models
can be safely exported without the ``torchscript`` flag.

Dummy inputs and standard lengths
-----------------------------------------------------------------------------------------------------------------------

The dummy inputs are used to do a model forward pass. While the inputs' values are propagating through the layers,
Pytorch keeps track of the different operations executed on each tensor. These recorded operations are then used to
create the "trace" of the model.

The trace is created relatively to the inputs' dimensions. It is therefore constrained by the dimensions of the dummy
input, and will not work for any other sequence length or batch size. When trying with a different size, an error such
as:

``The expanded size of the tensor (3) must match the existing size (7) at non-singleton dimension 2``

will be raised. It is therefore recommended to trace the model with a dummy input size at least as large as the largest
input that will be fed to the model during inference. Padding can be performed to fill the missing values. As the model
will have been traced with a large input size however, the dimensions of the different matrix will be large as well,
resulting in more calculations.

It is recommended to be careful of the total number of operations done on each input and to follow performance closely
when exporting varying sequence-length models.

Using TorchScript in Python
-----------------------------------------------------------------------------------------------------------------------

Below is an example, showing how to save, load models as well as how to use the trace for inference.

Saving a model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This snippet shows how to use TorchScript to export a ``BertModel``. Here the ``BertModel`` is instantiated according
to a ``BertConfig`` class and then saved to disk under the filename ``traced_bert.pt``

.. code-block:: python

    from transformers import BertModel, BertTokenizer, BertConfig
    import torch

    enc = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenizing input text
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = enc.tokenize(text)

    # Masking one of the input tokens
    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'
    indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    # Creating a dummy input
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    dummy_input = [tokens_tensor, segments_tensors]

    # Initializing the model with the torchscript flag
    # Flag set to True even though it is not necessary as this model does not have an LM Head.
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, torchscript=True)

    # Instantiating the model
    model = BertModel(config)

    # The model needs to be in evaluation mode
    model.eval()

    # If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
    model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)

    # Creating the trace
    traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
    torch.jit.save(traced_model, "traced_bert.pt")

Loading a model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This snippet shows how to load the ``BertModel`` that was previously saved to disk under the name ``traced_bert.pt``.
We are re-using the previously initialised ``dummy_input``.

.. code-block:: python

    loaded_model = torch.jit.load("traced_bert.pt")
    loaded_model.eval()

    all_encoder_layers, pooled_output = loaded_model(*dummy_input)

Using a traced model for inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using the traced model for inference is as simple as using its ``__call__`` dunder method:

.. code-block:: python

    traced_model(tokens_tensor, segments_tensors)
