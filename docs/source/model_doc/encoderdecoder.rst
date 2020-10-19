Encoder Decoder Models
-----------------------------------------------------------------------------------------------------------------------

The :class:`~transformers.EncoderDecoderModel` can be used to initialize a sequence-to-sequence model with any
pretrained autoencoding model as the encoder and any pretrained autoregressive model as the decoder.

The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation tasks
was shown in `Leveraging Pre-trained Checkpoints for Sequence Generation Tasks <https://arxiv.org/abs/1907.12461>`__ by
Sascha Rothe, Shashi Narayan, Aliaksei Severyn.

After such an :class:`~transformers.EncoderDecoderModel` has been trained/fine-tuned, it can be saved/loaded just like
any other models (see the examples for more information).

An application of this architecture could be to leverage two pretrained :class:`~transformers.BertModel` as the encoder
and decoder for a summarization model as was shown in: `Text Summarization with Pretrained Encoders
<https://arxiv.org/abs/1908.08345>`__ by Yang Liu and Mirella Lapata. 


EncoderDecoderConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.EncoderDecoderConfig
    :members:


EncoderDecoderModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.EncoderDecoderModel
    :members: forward
