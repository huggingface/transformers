Encoder Decoder Models
------------------------

This class can wrap an encoder model, such as ``BertModel`` and a decoder modeling with a language modeling head, such as ``BertForMaskedLM`` into a encoder-decoder model.

The ``EncoderDecoderModel`` class allows to instantiate a encoder decoder model using the ``from_encoder_decoder_pretrain`` class method taking a pretrained encoder and pretrained decoder model as an input. 
The ``EncoderDecoderModel`` is saved using the standard ``save_pretrained()`` method and can also again be loaded using the standard ``from_pretrained()`` method. 

An application of this architecture could be *summarization* using two pretrained Bert models as is shown in the paper: `Text Summarization with Pretrained Encoders <https://arxiv.org/abs/1910.13461>`_ by Yang Liu and Mirella Lapata. 


``EncoderDecoderConfig``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.EncoderDecoderConfig
    :members:


``EncoderDecoderModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.EncoderDecoderModel
    :members:
