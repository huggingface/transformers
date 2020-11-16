# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_sentencepiece_available, is_tf_available, is_torch_available
from .configuration_marian import MarianConfig


if is_sentencepiece_available():
    from .tokenization_marian import MarianTokenizer

if is_torch_available():
    from .modeling_marian import MarianMTModel

if is_tf_available():
    from .modeling_tf_marian import TFMarianMTModel
