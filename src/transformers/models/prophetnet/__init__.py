# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_torch_available
from .configuration_prophetnet import PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ProphetNetConfig
from .tokenization_prophetnet import ProphetNetTokenizer


if is_torch_available():
    from .modeling_prophetnet import (
        PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST,
        ProphetNetDecoder,
        ProphetNetEncoder,
        ProphetNetForCausalLM,
        ProphetNetForConditionalGeneration,
        ProphetNetModel,
        ProphetNetPreTrainedModel,
    )
