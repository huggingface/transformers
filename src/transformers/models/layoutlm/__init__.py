# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_tokenizers_available, is_torch_available
from .configuration_layoutlm import LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP, LayoutLMConfig
from .tokenization_layoutlm import LayoutLMTokenizer


if is_tokenizers_available():
    from .tokenization_layoutlm_fast import LayoutLMTokenizerFast

if is_torch_available():
    from .modeling_layoutlm import (
        LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,
        LayoutLMForMaskedLM,
        LayoutLMForTokenClassification,
        LayoutLMModel,
    )
