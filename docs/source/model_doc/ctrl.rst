CTRL
----------------------------------------------------

CTRL model was proposed in `CTRL: A Conditional Transformer Language Model for Controllable Generation <https://arxiv.org/abs/1909.05858>`_
by Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and Richard Socher.
It's a causal (unidirectional) transformer pre-trained using language modeling on a very large
corpus of ~140 GB of text data with the first token reserved as a control code (such as Links, Books, Wikipedia etc.).

The abstract from the paper is the following:

*Large-scale language models show promising text generation capabilities, but users cannot easily control particular
aspects of the generated text. We release CTRL, a 1.63 billion-parameter conditional transformer language model,
trained to condition on control codes that govern style, content, and task-specific behavior. Control codes were
derived from structure that naturally co-occurs with raw text, preserving the advantages of unsupervised learning
while providing more explicit control over text generation. These codes also allow CTRL to predict which parts of
the training data are most likely given a sequence. This provides a potential method for analyzing large amounts
of data via model-based source attribution.*

Tips:

- CTRL makes use of control codes to generate text: it requires generations to be started by certain words, sentences
  or links to generate coherent text. Refer to the `original implementation <https://github.com/salesforce/ctrl>`__
  for more information.
- CTRL is a model with absolute position embeddings so it's usually advised to pad the inputs on
  the right rather than the left.
- CTRL was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next
  token in a sequence. Leveraging this feature allows CTRL to generate syntactically coherent text as
  it can be observed in the `run_generation.py` example script.
- The PyTorch models can take the `past` as input, which is the previously computed key/value attention pairs. Using
  this `past` value prevents the model from re-computing pre-computed values in the context of text generation.
  See `reusing the past in generative models <../quickstart.html#using-the-past>`_ for more information on the usage
  of this argument.


CTRLConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CTRLConfig
    :members:


CTRLTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CTRLTokenizer
    :members: save_vocabulary


CTRLModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CTRLModel
    :members:


CTRLLMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CTRLLMHeadModel
    :members:


TFCTRLModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFCTRLModel
    :members:


TFCTRLLMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFCTRLLMHeadModel
    :members:

