# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_tokenizers_available
from .tokenization_herbert import HerbertTokenizer, HERBERT_PRETRAINED_TOKENIZER_ARCHIVE_LIST


if is_tokenizers_available():
    from .tokenization_herbert_fast import HerbertTokenizerFast, HERBERT_PRETRAINED_TOKENIZER_ARCHIVE_LIST
