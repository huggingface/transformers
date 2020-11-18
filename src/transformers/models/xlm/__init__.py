# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_tf_available, is_torch_available
from .configuration_xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig
from .tokenization_xlm import XLMTokenizer


if is_torch_available():
    from .modeling_xlm import (
        XLM_PRETRAINED_MODEL_ARCHIVE_LIST,
        XLMForMultipleChoice,
        XLMForQuestionAnswering,
        XLMForQuestionAnsweringSimple,
        XLMForSequenceClassification,
        XLMForTokenClassification,
        XLMModel,
        XLMPreTrainedModel,
        XLMWithLMHeadModel,
    )

if is_tf_available():
    from .modeling_tf_xlm import (
        TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFXLMForMultipleChoice,
        TFXLMForQuestionAnsweringSimple,
        TFXLMForSequenceClassification,
        TFXLMForTokenClassification,
        TFXLMMainLayer,
        TFXLMModel,
        TFXLMPreTrainedModel,
        TFXLMWithLMHeadModel,
    )
