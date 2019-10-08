TorchScript
================================================

.. note::
    This is the very beginning of our experiments with TorchScript and we are still exploring its capabilities
    with variable-input-size models. It is a focus of interest to us and we will deepen our analysis in upcoming
    releases, with more code examples, a more flexible implementation, and benchmarks comparing python-based codes
    with compiled TorchScript.


According to Pytorch's documentation: "TorchScript is a way to create serializable and optimizable models from PyTorch code".
Pytorch's two modules `JIT and TRACE <https://pytorch.org/docs/stable/jit.html>`_ allow the developer to export
their model to be re-used in other programs, such as efficiency-oriented C++ programs.

We have provided an interface that allows the export of `transformers` models to TorchScript so that they can
be reused in a different environment than a Pytorch-based python program. Here we explain how to use our models so that
they can be exported, and what to be mindful of when using these models with TorchScript.

Exporting a model needs two things:

* dummy inputs to execute a model forward pass.
* the model needs to be instantiated with the ``torchscript`` flag.

These necessities imply several things developers should be careful about. These are detailed below.


Implications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TorchScript flag and tied weights
------------------------------------------------
This flag is necessary because most of the language models in this repository have tied weights between their
``Embedding`` layer and their ``Decoding`` layer. TorchScript does not allow the export of models that have tied weights,
it is therefore necessary to untie the weights beforehand.

This implies that models instantiated with the ``torchscript`` flag have their ``Embedding`` layer and ``Decoding`` layer
separate, which means that they should not be trained down the line. Training would de-synchronize the two layers,
leading to unexpected results.

This is not the case for models that do not have a Language Model head, as those do not have tied weights. These models
can be safely exported without the ``torchscript`` flag.

Dummy inputs and standard lengths
------------------------------------------------

The dummy inputs are used to do a model forward pass. While the inputs' values are propagating through the layers,
Pytorch keeps track of the different operations executed on each tensor. These recorded operations are then used
to create the "trace" of the model.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below are examples of using the Python to save, load models as well as how to use the trace for inference.

Saving a model
------------------------------------------------

This snippet shows how to use TorchScript to export a ``BertModel``. Here the ``BertModel`` is instantiated
according to a ``BertConfig`` class and then saved to disk under the filename ``traced_bert.pt``

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
------------------------------------------------

This snippet shows how to load the ``BertModel`` that was previously saved to disk under the name ``traced_bert.pt``.
We are re-using the previously initialised ``dummy_input``.

.. code-block:: python

    loaded_model = torch.jit.load("traced_model.pt")
    loaded_model.eval()

    all_encoder_layers, pooled_output = loaded_model(dummy_input)

Using a traced model for inference
------------------------------------------------

Using the traced model for inference is as simple as using its ``__call__`` dunder method:

.. code-block:: python

    traced_model(tokens_tensor, segments_tensors)
