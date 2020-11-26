# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_flax_available, is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_mpnet import MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP, MPNetConfig
from .tokenization_mpnet import MPNetTokenizer


if is_tokenizers_available():
    from .tokenization_mpnet_fast import MPNetTokenizerFast

if is_torch_available():
    from .modeling_mpnet import (
        MPNET_PRETRAINED_MODEL_ARCHIVE_LIST,
        MPNetForMaskedLM,
        MPNetForMultipleChoice,
        MPNetForQuestionAnswering,
        MPNetForSequenceClassification,
        MPNetForTokenClassification,
        MPNetLayer,
        MPNetModel,
        MPNetPreTrainedModel,
    )

if is_tf_available():
    from .modeling_tf_mpnet import (
        TF_MPNET_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFMPNetEmbeddings,
        TFMPNetForMaskedLM,
        TFMPNetForMultipleChoice,
        TFMPNetForQuestionAnswering,
        TFMPNetForSequenceClassification,
        TFMPNetForTokenClassification,
        TFMPNetMainLayer,
        TFMPNetModel,
        TFMPNetPreTrainedModel,
    )
