# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_sentencepiece_available, is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_pegasus import PegasusConfig


if is_sentencepiece_available():
    from .tokenization_pegasus import PegasusTokenizer

if is_tokenizers_available():
    from .tokenization_pegasus_fast import PegasusTokenizerFast

if is_torch_available():
    from .modeling_pegasus import PegasusForConditionalGeneration

if is_tf_available():
    from .modeling_tf_pegasus import TFPegasusForConditionalGeneration
