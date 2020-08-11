Pegasus
----------------------------------------------------
**DISCLAIMER:** If you see something strange,
file a `Github Issue <https://github.com/huggingface/transformers/issues/new?assignees=sshleifer&labels=&template=bug-report.md&title>`__ and assign
@sshleifer.


Overview
~~~~~~~~~~~~~~~~~~~~~

The Pegasus model was `proposed <https://arxiv.org/abs/1910.13461>`_ by Jingqing Zhang, Yao Zhao, Mohammad Saleh and Peter J. Liu on Dec 18, 2019.
According to the abstract,

- Pegasus' pretraining task is intentionally similar to summarization: important sentences are removed/masked from an input document and are generated together as one output sequence from the remaining sentences, similar to an extractive summary.
- Pegasus achieves SOTA summarization performance on all 12 downstream tasks, as measured by ROUGE and human eval.

The Authors' code can be found `here <https://github.com/google-research/pegasus>`_


Checkpoints
~~~~~~~~~~~
The `checkpoints <https://huggingface.co/models?search=pegasus>`_ all checkpoints are finetuned for summarization, besides ``pegasus-large``, whence the other checkpoints are finetuned.
- Each checkpoint is 2.2 GB on disk and 568M parameters.
- FP16 is not supported (help/ideas on this appreciated!).
- Summarizing xsum in fp32 takes about 400ms/sample, with default parameters on a v100 GPU.
- For XSUM, The paper reports rouge1,rouge2, rougeL of paper: 47.21/24.56/39.25. As of Aug 9, this port scores 46.91/24.34/39.1.
The gap is likely because of different alpha/length_penalty implementations in beam search.


Implementation Notes
~~~~~~~~~~~~~~~~~~~~

- All models are transformer encoder-decoders with 16 layers in each component.
- The implementation is completely inherited from ``BartForConditionalGeneration``
- Some key configuration differences:

    - static, sinusoidal position embeddings
    - no ``layernorm_embedding`` (``PegasusConfig.normalize_embedding=False``)
    - the model starts generating with pad_token_id (which has 0 token_embedding) as the prefix.
    - ``num_beams=8``
- All pretrained pegasus checkpoints are the same besides three attributes: ``tokenizer.model_max_length`` (max input size),  ``max_length`` (max num tokens to generate) and ``length_penalty``
- Code to convert checkpoints trained in the author's `repo <https://github.com/google-research/pegasus>`_ can be found in ``convert_pegasus_tf_to_pytorch.py``





Usage Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    src_text = [
        """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""
    ]

    model_name = 'google/pegasus/xsum
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    batch = tokenizer(src_text, truncation=True, padding='longest')
    translated = model.generate(**batch)
    tgt_text = tokenizer.decode_batch(t, skip_special_tokens=True)

PegasusForConditionalGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model implementation inherits completely from ``BartForConditionalGeneration``
Available models are listed at `Model List <https://huggingface.co/models?search=pegasus>`__
This class inherits all functionality from ``BartForConditionalGeneration``, see that page for method signatures.

.. autoclass:: transformers.PegasusForConditionalGeneration
    :members: forward

PegasusConfig
~~~~~~~~~~~~~~~~~~~
.. autoclass:: transformers.PegasusConfig
    :members:


PegasusTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
warning: ``add_tokens`` does not work at the moment.

.. autoclass:: transformers.PegasusTokenizer
    :members: __call__, prepare_seq2seq_batch



