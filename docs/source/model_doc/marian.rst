MarianMT
-----------------------------------------------------------------------------------------------------------------------

**Bugs:** If you see something strange, file a `Github Issue
<https://github.com/huggingface/transformers/issues/new?assignees=sshleifer&labels=&template=bug-report.md&title>`__
and assign @sshleifer.

Translations should be similar, but not identical to, output in the test set linked to in each model card.

Implementation Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Each model is about 298 MB on disk, there are more than 1,000 models.
- The list of supported language pairs can be found `here <https://huggingface.co/Helsinki-NLP>`__.
- Models were originally trained by `Jörg Tiedemann
  <https://researchportal.helsinki.fi/en/persons/j%C3%B6rg-tiedemann>`__ using the `Marian
  <https://marian-nmt.github.io/>`__ C++ library, which supports fast training and translation.
- All models are transformer encoder-decoders with 6 layers in each component. Each model's performance is documented
  in a model card.
- The 80 opus models that require BPE preprocessing are not supported.
- The modeling code is the same as :class:`~transformers.BartForConditionalGeneration` with a few minor modifications:

    - static (sinusoid) positional embeddings (:obj:`MarianConfig.static_position_embeddings=True`)
    - a new final_logits_bias (:obj:`MarianConfig.add_bias_logits=True`)
    - no layernorm_embedding (:obj:`MarianConfig.normalize_embedding=False`)
    - the model starts generating with :obj:`pad_token_id` (which has 0 as a token_embedding) as the prefix (Bart uses
      :obj:`<s/>`),
- Code to bulk convert models can be found in ``convert_marian_to_pytorch.py``.

Naming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- All model names use the following format: :obj:`Helsinki-NLP/opus-mt-{src}-{tgt}`
- The language codes used to name models are inconsistent. Two digit codes can usually be found `here
  <https://developers.google.com/admin-sdk/directory/v1/languages>`__, three digit codes require googling "language
  code {code}".
- Codes formatted like :obj:`es_AR` are usually :obj:`code_{region}`. That one is Spanish from Argentina.


Multilingual Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All model names use the following format: :obj:`Helsinki-NLP/opus-mt-{src}-{tgt}`:

    - If :obj:`src` is in all caps, the model supports multiple input languages, you can figure out which ones by
      looking at the model card, or the Group Members `mapping
      <https://gist.github.com/sshleifer/6d20e7761931b08e73c3219027b97b8a>`_ .
    - If :obj:`tgt` is in all caps, the model can output multiple languages, and you should specify a language code by
      prepending the desired output language to the :obj:`src_text`.
    - You can see a tokenizer's supported language codes in ``tokenizer.supported_language_codes``

Example of translating english to many romance languages, using language codes:

.. code-block:: python

    from transformers import MarianMTModel, MarianTokenizer
    src_text = [
        '>>fr<< this is a sentence in english that we want to translate to french',
        '>>pt<< This should go to portuguese',
        '>>es<< And this to Spanish'
    ]

    model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    print(tokenizer.supported_language_codes)
    model = MarianMTModel.from_pretrained(model_name)
    translated = model.generate(**tokenizer.prepare_seq2seq_batch(src_text))
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    # ["c'est une phrase en anglais que nous voulons traduire en français",
    # 'Isto deve ir para o português.',
    # 'Y esto al español']

Sometimes, models were trained on collections of languages that do not resolve to a group. In this case, _ is used as a
separator for src or tgt, as in :obj:`Helsinki-NLP/opus-mt-en_el_es_fi-en_el_es_fi`. These still require language
codes.

There are many supported regional language codes, like :obj:`>>es_ES<<` (Spain) and :obj:`>>es_AR<<` (Argentina), that
do not seem to change translations. I have not found these to provide different results than just using :obj:`>>es<<`.

For example:

    - `Helsinki-NLP/opus-mt-NORTH_EU-NORTH_EU`: translates from all NORTH_EU languages (see `mapping
      <https://gist.github.com/sshleifer/6d20e7761931b08e73c3219027b97b8a>`_) to all NORTH_EU languages. Use a special
      language code like :obj:`>>de<<` to specify output language.
    - `Helsinki-NLP/opus-mt-ROMANCE-en`: translates from many romance languages to english, no codes needed since there
      is only one target language.



.. code-block:: python

    GROUP_MEMBERS = {
     'ZH': ['cmn', 'cn', 'yue', 'ze_zh', 'zh_cn', 'zh_CN', 'zh_HK', 'zh_tw', 'zh_TW', 'zh_yue', 'zhs', 'zht', 'zh'],
     'ROMANCE': ['fr', 'fr_BE', 'fr_CA', 'fr_FR', 'wa', 'frp', 'oc', 'ca', 'rm', 'lld', 'fur', 'lij', 'lmo', 'es', 'es_AR', 'es_CL', 'es_CO', 'es_CR', 'es_DO', 'es_EC', 'es_ES', 'es_GT', 'es_HN', 'es_MX', 'es_NI', 'es_PA', 'es_PE', 'es_PR', 'es_SV', 'es_UY', 'es_VE', 'pt', 'pt_br', 'pt_BR', 'pt_PT', 'gl', 'lad', 'an', 'mwl', 'it', 'it_IT', 'co', 'nap', 'scn', 'vec', 'sc', 'ro', 'la'],
     'NORTH_EU': ['de', 'nl', 'fy', 'af', 'da', 'fo', 'is', 'no', 'nb', 'nn', 'sv'],
     'SCANDINAVIA': ['da', 'fo', 'is', 'no', 'nb', 'nn', 'sv'],
     'SAMI': ['se', 'sma', 'smj', 'smn', 'sms'],
     'NORWAY': ['nb_NO', 'nb', 'nn_NO', 'nn', 'nog', 'no_nb', 'no'],
     'CELTIC': ['ga', 'cy', 'br', 'gd', 'kw', 'gv']
    }

Code to see available pretrained models:

.. code-block:: python

    from transformers.hf_api import HfApi
    model_list = HfApi().model_list()
    org = "Helsinki-NLP"
    model_ids = [x.modelId for x in model_list if x.modelId.startswith(org)]
    suffix = [x.split('/')[1] for x in model_ids]
    multi_models = [f'{org}/{s}' for s in suffix if s != s.lower()]


MarianConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MarianConfig
    :members:


MarianTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MarianTokenizer
    :members: prepare_seq2seq_batch


MarianMTModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MarianMTModel


TFMarianMTModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFMarianMTModel
