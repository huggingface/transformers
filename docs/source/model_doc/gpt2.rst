OpenAI GPT2
----------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~

OpenAI GPT-2 model was proposed in
`Language Models are Unsupervised Multitask Learners`_
by Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever**.
It's a causal (unidirectional) transformer pre-trained using  language modeling on a very large
corpus of ~40 GB of text data.

The abstract from the paper is the following:

*GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset[1]
of 8 million web pages. GPT-2 is trained with a simple objective: predict the next word, given all of the previous
words within some text. The diversity of the dataset causes this simple goal to contain naturally occurring
demonstrations of many tasks across diverse domains. GPT-2 is a direct scale-up of GPT, with more than 10X
the parameters and trained on more than 10X the amount of data.*

Tips:

- GPT-2 is a model with absolute position embeddings so it's usually advised to pad the inputs on
  the right rather than the left.
- GPT-2 was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next
  token in a sequence. Leveraging this feature allows GPT-2 to generate syntactically coherent text as
  it can be observed in the `run_generation.py` example script.
- The PyTorch models can take the `past` as input, which is the previously computed key/value attention pairs. Using
  this `past` value prevents the model from re-computing pre-computed values in the context of text generation.
  See `reusing the past in generative models <../quickstart.html#using-the-past>`_ for more information on the usage
  of this argument.

`Write With Transformer <https://transformer.huggingface.co/doc/gpt2-large>`__ is a webapp created and hosted by
Hugging Face showcasing the generative capabilities of several models. GPT-2 is one of them and is available in five
different sizes: small, medium, large, xl and a distilled version of the small checkpoint: distilgpt-2.


GPT2Config
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPT2Config
    :members:


GPT2Tokenizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPT2Tokenizer
    :members:


GPT2Model
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPT2Model
    :members:


GPT2LMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPT2LMHeadModel
    :members:


GPT2DoubleHeadsModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPT2DoubleHeadsModel
    :members:


TFGPT2Model
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFGPT2Model
    :members:


TFGPT2LMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFGPT2LMHeadModel
    :members:


TFGPT2DoubleHeadsModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFGPT2DoubleHeadsModel
    :members:
