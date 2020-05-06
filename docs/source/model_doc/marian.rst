MarianMTModel
----------------------------------------------------
**DISCLAIMER:** If you see something strange,
file a `Github Issue <https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title>`__ and assign
@sshleifer
These models are for machine translation. The list of supported language pairs can be found `here <https://huggingface.co/Helsinki-NLP>`__.

Opus Project
~~~~~~~~~~~~
The 1,000+ models were originally trained by `Jörg Tiedemann <https://researchportal.helsinki.fi/en/persons/j%C3%B6rg-tiedemann>`__ using the `Marian <https://marian-nmt.github.io/>`_ C++ library, which supports fast training and translation.
All models are transformer encoder-decoders with 6 layers in each component. Each model's performance is documented in a model card.

Implementation Notes
~~~~~~~~~~~~~~~~~~~~
- each model is about 298 MB on disk, there are 1,000+ models.
- Models are named with the following patter 'Helsinki-NLP/opus-mt-{src_langs}-{targ_langs}'. If there are multiple source or target languages they are joined by a '+' symbol.
- the 80 opus models that require BPE preprocessing are not supported.
- There is an outstanding issue w.r.t multilingual models and language codes.
- The modeling code is the same as ``BartModel`` with a few minor modifications:
    - static (sinusoid) positional embeddings (``MarianConfig.static_position_embeddings=True``)
    - a new final_logits_bias (``MarianConfig.add_bias_logits=True``)
    - no layernorm_embedding (``MarianConfig.normalize_embedding=False``)
    - the model starts generating with pad_token_id (which has 0 token_embedding) as the prefix. (Bart uses <s/>)
- Code to bulk convert models can be found in ``convert_marian_to_pytorch.py``


MarianMTModel
~~~~~~~~~~~~~
This class inherits all functionality from ``BartForConditionalGeneration``, see that page for method signatures.

MarianTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MarianTokenizer
    :members: prepare_translation_batch
