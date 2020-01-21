Transformer XL
----------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~

The Transformer-XL model was proposed in
`Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context <https://arxiv.org/abs/1901.02860>`__
by Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
It's a causal (uni-directional) transformer with relative positioning (sinusoïdal) embeddings which can reuse
previously computed hidden-states to attend to longer context (memory).
This model also uses adaptive softmax inputs and outputs (tied).

The abstract from the paper is the following:

*Transformers have a potential of learning longer-term dependency, but are limited by a fixed-length context in the
setting of language modeling. We propose a novel neural architecture Transformer-XL that enables learning dependency
beyond a fixed length without disrupting temporal coherence. It consists of a segment-level recurrence mechanism and
a novel positional encoding scheme. Our method not only enables capturing longer-term dependency, but also resolves
the context fragmentation problem. As a result, Transformer-XL learns dependency that is 80% longer than RNNs and
450% longer than vanilla Transformers, achieves better performance on both short and long sequences, and is up
to 1,800+ times faster than vanilla Transformers during evaluation. Notably, we improve the state-of-the-art results
of bpc/perplexity to 0.99 on enwiki8, 1.08 on text8, 18.3 on WikiText-103, 21.8 on One Billion Word, and 54.5 on
Penn Treebank (without finetuning). When trained only on WikiText-103, Transformer-XL manages to generate reasonably
coherent, novel text articles with thousands of tokens.*

Tips:

- Transformer-XL uses relative sinusoidal positional embeddings. Padding can be done on the left or on the right.
  The original implementation trains on SQuAD with padding on the left, therefore the padding defaults are set to left.
- Transformer-XL is one of the few models that has no sequence length limit.


TransfoXLConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TransfoXLConfig
    :members:


TransfoXLTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TransfoXLTokenizer
    :members:


TransfoXLModel
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TransfoXLModel
    :members:


TransfoXLLMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TransfoXLLMHeadModel
    :members:


TFTransfoXLModel
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFTransfoXLModel
    :members:


TFTransfoXLLMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFTransfoXLLMHeadModel
    :members:
