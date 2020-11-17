# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_distilbert import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DistilBertConfig
from .tokenization_distilbert import DistilBertTokenizer


if is_tokenizers_available():
    from .tokenization_distilbert_fast import DistilBertTokenizerFast

if is_torch_available():
    from .modeling_distilbert import (
        DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        DistilBertForMaskedLM,
        DistilBertForMultipleChoice,
        DistilBertForQuestionAnswering,
        DistilBertForSequenceClassification,
        DistilBertForTokenClassification,
        DistilBertModel,
        DistilBertPreTrainedModel,
    )

if is_tf_available():
    from .modeling_tf_distilbert import (
        TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFDistilBertForMaskedLM,
        TFDistilBertForMultipleChoice,
        TFDistilBertForQuestionAnswering,
        TFDistilBertForSequenceClassification,
        TFDistilBertForTokenClassification,
        TFDistilBertMainLayer,
        TFDistilBertModel,
        TFDistilBertPreTrainedModel,
    )
