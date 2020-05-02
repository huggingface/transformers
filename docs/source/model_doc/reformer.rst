Reformer
----------------------------------------------------
**DISCLAIMER:** This model is still a work in progress, if you see something strange,
file a `Github Issue <https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title>`_

Overview
~~~~~
The Reformer model was presented in `Reformer: The Efficient Transformer <https://https://arxiv.org/abs/2001.04451.pdf>`_ by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.
Here the abstract: 

*Large Transformer models routinely achieve state-of-the-art results on a number of tasks but training these models can be prohibitively costly, especially on long sequences. We introduce two techniques to improve the efficiency of Transformers. For one, we replace dot-product attention by one that uses locality-sensitive hashing, changing its complexity from O(L^2) to O(Llog(L)), where L is the length of the sequence. Furthermore, we use reversible residual layers instead of the standard residuals, which allows storing activations only once in the training process instead of N times, where N is the number of layers. The resulting model, the Reformer, performs on par with Transformer models while being much more memory-efficient and much faster on long sequences.*

The Authors' code can be found `here <https://github.com/google/trax/tree/master/trax/models/reformer>`_ .

Axial Positional Encodings
~~~~~~~~~~~~~~~~~~~~
Axial Positional Encodings were first implemented in Google's `trax library <https://github.com/google/trax/blob/4d99ad4965bab1deba227539758d59f0df0fef48/trax/layers/research/position_encodings.py#L29>`_ and developed by the authors of this model's paper. In Models that are treating very long input sequences, the conventional position id encodings store an embedings vector of size :math:`d` (``config.hidden_size``) for every position :math:`i, \ldots, n_s`, with :math:`n_s` being ``config.max_embedding_size``. *E.g.*, having a sequence length of :math:`n_s = 2^{19} \approx 0.5M` and a ``config.hidden_size`` of :math:`d = 2^{10} \approx 1000` would result in a position encoding matrix:

.. math::
    X_{i,j}, \text{ with } i \in \left[1,\ldots, d\right] \text{ and } j \in \left[1,\ldots, n_s\right] 

which alone has over 500M parameters to store. Axial positional encodings factorize :math:`X_{i,j}` into two matrices: 

.. math::
    X^{1}_{i,j^1}, \text{ with } i \in \left[1,\ldots, d^1\right] \text{ and } j^1 \in \left[1,\ldots, n_s^1\right] 

and 

.. math::
    X^{2}_{i,j^2}, \text{ with } i \in \left[1,\ldots, d^2\right] \text{ and } j^2 \in \left[1,\ldots, n_s^2\right] 

with:

.. math::
    d = d^1 + d^2 \text{ and } n_s = n_s^1 \times n_s^2 .

Therefore the following holds:

.. math::
    X_{i,j} = \begin{cases}
                X^{1}_{i, k}, & \text{if }\ i < d^1 \text{ with } k = j \mod n_s^1 \\
                X^{2}_{i - d^1, l}, & \text{if } i \ge d^1 \text{ with } l = \lfloor\frac{j}{n_s^1}\rfloor
              \end{cases}

Intuitively, this means that a position embedding vector :math:`x_j \in \mathbb{R}^{d}` is now the composition of two factorized embedding vectors: :math:`x^1_{k, l} + x^2_{l, k}`, where as the ``config.max_embedding_size`` dimension :math:`j` is factorized into :math:`k \text{ and } l`.
This design ensures that each position embedding vector :math:`x_j` is unique.

Using the above example again, axial position encoding with :math:`d^1 = 2^5, d^2 = 2^5, n_s^1 = 2^9, n_s^2 = 2^10` can drastically reduced the number of parameters to :math:`2^14 + 2^15 \approx 49000` parameters.

In praxis, the parameter ``config.axial_pos_embds_dim`` is set to ``list(d^1, d^2)`` which sum has to be equal to ``config.hidden_size`` and ``config.axial_pos_shape`` is set to ``list(n_s^1, n_s^2)`` and which product has to be equal to ``config.max_embedding_size`` which during training has to be equal to the ``sequence length`` of the ``input_ids``.



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
