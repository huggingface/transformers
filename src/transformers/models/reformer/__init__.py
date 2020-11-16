# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_sentencepiece_available, is_tokenizers_available, is_torch_available
from .configuration_reformer import REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, ReformerConfig


if is_sentencepiece_available():
    from .tokenization_reformer import ReformerTokenizer

if is_tokenizers_available():
    from .tokenization_reformer_fast import ReformerTokenizerFast

if is_torch_available():
    from .modeling_reformer import (
        REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
        ReformerAttention,
        ReformerForMaskedLM,
        ReformerForQuestionAnswering,
        ReformerForSequenceClassification,
        ReformerLayer,
        ReformerModel,
        ReformerModelWithLMHead,
    )
