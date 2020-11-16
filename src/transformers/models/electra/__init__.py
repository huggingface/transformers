# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig
from .tokenization_electra import ElectraTokenizer


if is_tokenizers_available():
    from .tokenization_electra_fast import ElectraTokenizerFast

if is_torch_available():
    from .modeling_electra import (
        ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
        ElectraForMaskedLM,
        ElectraForMultipleChoice,
        ElectraForPreTraining,
        ElectraForQuestionAnswering,
        ElectraForSequenceClassification,
        ElectraForTokenClassification,
        ElectraModel,
        ElectraPreTrainedModel,
        load_tf_weights_in_electra,
    )

if is_tf_available():
    from .modeling_tf_electra import (
        TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFElectraForMaskedLM,
        TFElectraForMultipleChoice,
        TFElectraForPreTraining,
        TFElectraForQuestionAnswering,
        TFElectraForSequenceClassification,
        TFElectraForTokenClassification,
        TFElectraModel,
        TFElectraPreTrainedModel,
    )
