# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_torch_available
from .configuration_tapas import TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP, TapasConfig
from .tokenization_tapas import TapasTokenizer


if is_torch_available():
    from .modeling_tapas import (
        TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST,
        TapasForMaskedLM,
        TapasForQuestionAnswering,
        TapasForSequenceClassification,
        TapasModel,
    )
