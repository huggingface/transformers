Reformer
----------------------------------------------------
**DISCLAIMER:** This model is still a work in progress, if you see something strange,
file a `Github Issue <https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title>`_

Overview
~~~~~
The Reformer model was presented in `Reformer: The Efficient Transformer <https://https://arxiv.org/abs/2001.04451.pdf>`_ by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.
Here the abstract: 

*Large Transformer models routinely achieve state-of-the-art results on a number of tasks but training these models can be prohibitively costly, especially on long sequences. We introduce two techniques to improve the efficiency of Transformers. For one, we replace dot-product attention by one that uses locality-sensitive hashing, changing its complexity from O(L^2) to O(Llog(L)), where L is the length of the sequence. Furthermore, we use reversible residual layers instead of the standard residuals, which allows storing activations only once in the training process instead of N times, where N is the number of layers. The resulting model, the Reformer, performs on par with Transformer models while being much more memory-efficient and much faster on long sequences.*

The Authors' code can be found `here https://github.com/google/trax/tree/master/trax/models/reformer>`_ .

Axial Positional Encodings
~~~~~~~~~~~~~~~~~~~~
TODO

LSH Sensitive Hashing Self Attention
~~~~~~~~~~~~~~~~~~~~
TODO

Local Sensitive Hashing Self Attention
~~~~~~~~~~~~~~~~~~~~
TODO

Training
~~~~~~~~~~~~~~~~~~~~
TODO

Tips
~~~~~~~~~~~~~~~~~~~~
TODO

ReformerConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ReformerConfig
    :members:


ReformerTokenizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ReformerTokenizer
    :members: 


ReformerModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ReformerModel
    :members:


ReformerModelWithLMHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ReformerModelWithLMHead
    :members:
