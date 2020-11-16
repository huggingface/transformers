# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_sentencepiece_available, is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CamembertConfig


if is_sentencepiece_available():
    from .tokenization_camembert import CamembertTokenizer

if is_tokenizers_available():
    from .tokenization_camembert_fast import CamembertTokenizerFast

if is_torch_available():
    from .modeling_camembert import (
        CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        CamembertForCausalLM,
        CamembertForMaskedLM,
        CamembertForMultipleChoice,
        CamembertForQuestionAnswering,
        CamembertForSequenceClassification,
        CamembertForTokenClassification,
        CamembertModel,
    )

if is_tf_available():
    from .modeling_tf_camembert import (
        TF_CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFCamembertForMaskedLM,
        TFCamembertForMultipleChoice,
        TFCamembertForQuestionAnswering,
        TFCamembertForSequenceClassification,
        TFCamembertForTokenClassification,
        TFCamembertModel,
    )
