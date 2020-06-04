ELECTRA
----------------------------------------------------

The ELECTRA model was proposed in the paper.
`ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators <https://openreview.net/pdf?id=r1xMH1BtvB>`__.
ELECTRA is a new pre-training approach which trains two transformer models: the generator and the discriminator. The
generator's role is to replace tokens in a sequence, and is therefore trained as a masked language model. The discriminator,
which is the model we're interested in, tries to identify which tokens were replaced by the generator in the sequence.

The abstract from the paper is the following:

*Masked language modeling (MLM) pre-training methods such as BERT corrupt
the input by replacing some tokens with [MASK] and then train a model to
reconstruct the original tokens. While they produce good results when transferred
to downstream NLP tasks, they generally require large amounts of compute to be
effective. As an alternative, we propose a more sample-efficient pre-training task
called replaced token detection. Instead of masking the input, our approach
corrupts it by replacing some tokens with plausible alternatives sampled from a small
generator network. Then, instead of training a model that predicts the original
identities of the corrupted tokens, we train a discriminative model that predicts
whether each token in the corrupted input was replaced by a generator sample
or not. Thorough experiments demonstrate this new pre-training task is more
efficient than MLM because the task is defined over all input tokens rather than
just the small subset that was masked out. As a result, the contextual representations
learned by our approach substantially outperform the ones learned by BERT
given the same model size, data, and compute. The gains are particularly strong
for small models; for example, we train a model on one GPU for 4 days that
outperforms GPT (trained using 30x more compute) on the GLUE natural language
understanding benchmark. Our approach also works well at scale, where it
performs comparably to RoBERTa and XLNet while using less than 1/4 of their
compute and outperforms them when using the same amount of compute.*

Tips:

- ELECTRA is the pre-training approach, therefore there is nearly no changes done to the underlying model: BERT. The
  only change is the separation of the embedding size and the hidden size -> The embedding size is generally smaller,
  while the hidden size is larger. An additional projection layer (linear) is used to project the embeddings from
  their embedding size to the hidden size. In the case where the embedding size is the same as the hidden size, no
  projection layer is used.
- The ELECTRA checkpoints saved using `Google Research's implementation <https://github.com/google-research/electra>`__
  contain both the generator and discriminator. The conversion script requires the user to name which model to export
  into the correct architecture. Once converted to the HuggingFace format, these checkpoints may be loaded into all
  available ELECTRA models, however. This means that the discriminator may be loaded in the `ElectraForMaskedLM` model,
  and the generator may be loaded in the `ElectraForPreTraining` model (the classification head will be randomly
  initialized as it doesn't exist in the generator).

The original code can be found `here <https://github.com/google-research/electra>`_.


ElectraConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ElectraConfig
    :members:


ElectraTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ElectraTokenizer
    :members:


ElectraTokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ElectraTokenizerFast
    :members:


ElectraModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ElectraModel
    :members:


ElectraForPreTraining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ElectraForPreTraining
    :members:


ElectraForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ElectraForMaskedLM
    :members:


ElectraForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ElectraForTokenClassification
    :members:


TFElectraModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFElectraModel
    :members:


TFElectraForPreTraining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFElectraForPreTraining
    :members:


TFElectraForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFElectraForMaskedLM
    :members:


TFElectraForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFElectraForTokenClassification
    :members:
