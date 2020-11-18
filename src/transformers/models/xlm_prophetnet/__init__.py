# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_sentencepiece_available, is_torch_available
from .configuration_xlm_prophetnet import XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMProphetNetConfig


if is_sentencepiece_available():
    from .tokenization_xlm_prophetnet import XLMProphetNetTokenizer

if is_torch_available():
    from .modeling_xlm_prophetnet import (
        XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST,
        XLMProphetNetDecoder,
        XLMProphetNetEncoder,
        XLMProphetNetForCausalLM,
        XLMProphetNetForConditionalGeneration,
        XLMProphetNetModel,
    )
