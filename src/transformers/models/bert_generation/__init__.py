# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_sentencepiece_available, is_torch_available
from .configuration_bert_generation import BertGenerationConfig


if is_sentencepiece_available():
    from .tokenization_bert_generation import BertGenerationTokenizer

if is_torch_available():
    from .modeling_bert_generation import (
        BertGenerationDecoder,
        BertGenerationEncoder,
        load_tf_weights_in_bert_generation,
    )
