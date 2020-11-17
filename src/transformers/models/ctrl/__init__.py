# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_tf_available, is_torch_available
from .configuration_ctrl import CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRLConfig
from .tokenization_ctrl import CTRLTokenizer


if is_torch_available():
    from .modeling_ctrl import CTRL_PRETRAINED_MODEL_ARCHIVE_LIST, CTRLLMHeadModel, CTRLModel, CTRLPreTrainedModel

if is_tf_available():
    from .modeling_tf_ctrl import (
        TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFCTRLLMHeadModel,
        TFCTRLModel,
        TFCTRLPreTrainedModel,
    )
