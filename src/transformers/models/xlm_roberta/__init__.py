# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_sentencepiece_available, is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_xlm_roberta import XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMRobertaConfig


if is_sentencepiece_available():
    from .tokenization_xlm_roberta import XLMRobertaTokenizer

if is_tokenizers_available():
    from .tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast

if is_torch_available():
    from .modeling_xlm_roberta import (
        XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        XLMRobertaForCausalLM,
        XLMRobertaForMaskedLM,
        XLMRobertaForMultipleChoice,
        XLMRobertaForQuestionAnswering,
        XLMRobertaForSequenceClassification,
        XLMRobertaForTokenClassification,
        XLMRobertaModel,
    )

if is_tf_available():
    from .modeling_tf_xlm_roberta import (
        TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFXLMRobertaForMaskedLM,
        TFXLMRobertaForMultipleChoice,
        TFXLMRobertaForQuestionAnswering,
        TFXLMRobertaForSequenceClassification,
        TFXLMRobertaForTokenClassification,
        TFXLMRobertaModel,
    )
