# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_torch_available
from .configuration_deberta import DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, DebertaConfig
from .tokenization_deberta import DebertaTokenizer


if is_torch_available():
    from .modeling_deberta import (
        DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        DebertaForSequenceClassification,
        DebertaModel,
        DebertaPreTrainedModel,
    )
