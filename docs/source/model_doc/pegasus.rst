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

    model_name = 'google/pegasus-xsum'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    batch = tokenizer.prepare_seq2seq_batch(src_text, truncation=True, padding='longest').to(torch_device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    assert tgt_text[0] == "California's largest electricity provider has turned off power to tens of thousands of customers."

PegasusForConditionalGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This class inherits all functionality from ``BartForConditionalGeneration``, see that page for method signatures.
Available models are listed at `Model List <https://huggingface.co/models?search=pegasus>`__

.. autoclass:: transformers.PegasusForConditionalGeneration
    :members:


PegasusConfig
~~~~~~~~~~~~~~~~~~~
This config fully inherits from ``BartConfig``, but pegasus uses different default values:
Up to date parameter values can be seen in `S3 <https://s3.amazonaws.com/models.huggingface.co/bert/google/pegasus-xsum/config.json>`_.
As of Aug 10, 2020, they are:

.. code-block:: python

    dict(
    vocab_size=96103,
    max_position_embeddings=512,
    d_model=1024,
    encoder_ffn_dim=4096,
    decoder_ffn_dim=4096,
    encoder_attention_heads=16,
    decoder_attention_heads=16,
    encoder_layers=16,
    decoder_layers=16,
    dropout=0.1,
    attention_dropout=0.1,
    activation_dropout=0.1,
    pad_token_id=0,
    eos_token_id=1,
    is_encoder_decoder=True,
    normalize_before=True,
    scale_embedding=True,
    normalize_embedding=False,
    add_final_layer_norm=True,
    static_position_embeddings=True,
    num_beams=8,
    activation_function="relu",
    )


PegasusTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
warning: ``add_tokens`` does not work at the moment.

.. autoclass:: transformers.PegasusTokenizer
    :members: __call__, prepare_seq2seq_batch



