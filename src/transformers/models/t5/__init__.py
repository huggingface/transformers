# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_sentencepiece_available, is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config


if is_sentencepiece_available():
    from .tokenization_t5 import T5Tokenizer

if is_tokenizers_available():
    from .tokenization_t5_fast import T5TokenizerFast

if is_torch_available():
    from .modeling_t5 import (
        T5_PRETRAINED_MODEL_ARCHIVE_LIST,
        T5ForConditionalGeneration,
        T5Model,
        T5PreTrainedModel,
        load_tf_weights_in_t5,
    )

if is_tf_available():
    from .modeling_tf_t5 import (
        TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFT5ForConditionalGeneration,
        TFT5Model,
        TFT5PreTrainedModel,
    )
