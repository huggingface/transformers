# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_funnel import FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP, FunnelConfig
from .tokenization_funnel import FunnelTokenizer


if is_tokenizers_available():
    from .tokenization_funnel_fast import FunnelTokenizerFast

if is_torch_available():
    from .modeling_funnel import (
        FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST,
        FunnelBaseModel,
        FunnelForMaskedLM,
        FunnelForMultipleChoice,
        FunnelForPreTraining,
        FunnelForQuestionAnswering,
        FunnelForSequenceClassification,
        FunnelForTokenClassification,
        FunnelModel,
        load_tf_weights_in_funnel,
    )

if is_tf_available():
    from .modeling_tf_funnel import (
        TF_FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFFunnelBaseModel,
        TFFunnelForMaskedLM,
        TFFunnelForMultipleChoice,
        TFFunnelForPreTraining,
        TFFunnelForQuestionAnswering,
        TFFunnelForSequenceClassification,
        TFFunnelForTokenClassification,
        TFFunnelModel,
    )
