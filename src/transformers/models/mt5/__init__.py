# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_sentencepiece_available, is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_mt5 import MT5Config


if is_sentencepiece_available():
    from ..t5.tokenization_t5 import T5Tokenizer

    MT5Tokenizer = T5Tokenizer

if is_tokenizers_available():
    from ..t5.tokenization_t5_fast import T5TokenizerFast

    MT5TokenizerFast = T5TokenizerFast

if is_torch_available():
    from .modeling_mt5 import MT5EncoderModel, MT5ForConditionalGeneration, MT5Model

if is_tf_available():
    from .modeling_tf_mt5 import TFMT5EncoderModel, TFMT5ForConditionalGeneration, TFMT5Model
