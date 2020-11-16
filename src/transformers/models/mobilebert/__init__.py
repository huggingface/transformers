# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_mobilebert import MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, MobileBertConfig
from .tokenization_mobilebert import MobileBertTokenizer


if is_tokenizers_available():
    from .tokenization_mobilebert_fast import MobileBertTokenizerFast

if is_torch_available():
    from .modeling_mobilebert import (
        MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        MobileBertForMaskedLM,
        MobileBertForMultipleChoice,
        MobileBertForNextSentencePrediction,
        MobileBertForPreTraining,
        MobileBertForQuestionAnswering,
        MobileBertForSequenceClassification,
        MobileBertForTokenClassification,
        MobileBertLayer,
        MobileBertModel,
        MobileBertPreTrainedModel,
        load_tf_weights_in_mobilebert,
    )

if is_tf_available():
    from .modeling_tf_mobilebert import (
        TF_MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFMobileBertForMaskedLM,
        TFMobileBertForMultipleChoice,
        TFMobileBertForNextSentencePrediction,
        TFMobileBertForPreTraining,
        TFMobileBertForQuestionAnswering,
        TFMobileBertForSequenceClassification,
        TFMobileBertForTokenClassification,
        TFMobileBertMainLayer,
        TFMobileBertModel,
        TFMobileBertPreTrainedModel,
    )
