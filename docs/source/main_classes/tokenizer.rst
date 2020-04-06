Tokenizer
----------------------------------------------------

The base class ``PreTrainedTokenizer`` implements the common methods for loading/saving a tokenizer either from a local file or directory, or from a pretrained tokenizer provided by the library (downloaded from HuggingFace's AWS S3 repository).

``PreTrainedTokenizer`` is the main entry point into tokenizers as it also implements the main methods for using all the tokenizers:

- tokenizing, converting tokens to ids and back and encoding/decoding,
- adding new tokens to the vocabulary in a way that is independant of the underlying structure (BPE, SentencePiece...),
- managing special tokens (adding them, assigning them to roles, making sure they are not split during tokenization)

``PreTrainedTokenizer``
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PreTrainedTokenizer
    :members:
