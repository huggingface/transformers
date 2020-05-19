Tokenizer
----------------------------------------------------

A tokenizer is in charge of preparing the inputs for a model. The library comprise tokenizers for all the models. Most of the tokenizers are available in two flavors: a full python implementation and a "Fast" implementation based on the Rust library `tokenizers`. The "Fast" implementations allows (1) a significant speed-up in particular when doing batched tokenization and (2) additional methods to map between the original string (character and words) and the token space (e.g. getting the index of the token comprising a given character or the span of characters corresponding to a given token). Currently no "Fast" implementation is available for the SentencePiece-based tokenizers (for T5, ALBERT, CamemBERT, XLMRoBERTa and XLNet models).

The base classes ``PreTrainedTokenizer`` and ``PreTrainedTokenizerFast`` implements the common methods for encoding string inputs in model inputs (see below) and instantiating/saving python and "Fast" tokenizers either from a local file or directory or from a pretrained tokenizer provided by the library (downloaded from HuggingFace's AWS S3 repository).

``PreTrainedTokenizer`` and ``PreTrainedTokenizerFast`` thus implements the main methods for using all the tokenizers:

- tokenizing (spliting strings in sub-word token strings), converting tokens strings to ids and back, and encoding/decoding (i.e. tokenizing + convert to integers),
- adding new tokens to the vocabulary in a way that is independant of the underlying structure (BPE, SentencePiece...),
- managing special tokens like mask, beginning-of-sentence, etc tokens (adding them, assigning them to attributes in the tokenizer for easy access and making sure they are not split during tokenization)

``BatchEncoding`` holds the output of the tokenizer's encoding methods (``encode_plus`` and ``batch_encode_plus``) and is derived from a Python dictionary. When the tokenizer is a pure python tokenizer, this class behave just like a standard python dictionary and hold the various model inputs computed by these methodes (``input_ids``, ``attention_mask``...). When the tokenizer is a "Fast" tokenizer (i.e. backed by HuggingFace tokenizers library), this class provides in addition several advanced alignement methods which can be used to map between the original string (character and words) and the token space (e.g. getting the index of the token comprising a given character or the span of characters corresponding to a given token).

``PreTrainedTokenizer``
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PreTrainedTokenizer
    :members:

``PreTrainedTokenizerFast``
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PreTrainedTokenizerFast
    :members:

``BatchEncoding``
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BatchEncoding
    :members:

``SpecialTokensMixin``
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SpecialTokensMixin
    :members:
