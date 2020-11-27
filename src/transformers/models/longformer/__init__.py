# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_longformer import LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, LongformerConfig
from .tokenization_longformer import LongformerTokenizer


if is_tokenizers_available():
    from .tokenization_longformer_fast import LongformerTokenizerFast

if is_torch_available():
    from .modeling_longformer import (
        LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
        LongformerForMaskedLM,
        LongformerForMultipleChoice,
        LongformerForQuestionAnswering,
        LongformerForSequenceClassification,
        LongformerForTokenClassification,
        LongformerModel,
        LongformerSelfAttention,
    )

if is_tf_available():
    from .modeling_tf_longformer import (
        TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFLongformerForMaskedLM,
        TFLongformerForMultipleChoice,
        TFLongformerForQuestionAnswering,
        TFLongformerForSequenceClassification,
        TFLongformerForTokenClassification,
        TFLongformerModel,
        TFLongformerSelfAttention,
    )
