# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import (
    is_flax_available,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
    is_tf_available
)

from .configuration import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig

if is_sentencepiece_available():
    from .tokenization_slow import AlbertTokenizer

if is_tokenizers_available():
    from .tokenization_fast import AlbertTokenizerFast

if is_torch_available():
    from .pt_model import (
        ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        AlbertForMaskedLM,
        AlbertForMultipleChoice,
        AlbertForPreTraining,
        AlbertForQuestionAnswering,
        AlbertForSequenceClassification,
        AlbertForTokenClassification,
        AlbertModel,
        AlbertPreTrainedModel,
        load_tf_weights_in_albert,
    )

if is_tf_available():
    from .tf_model import (
        TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFAlbertForMaskedLM,
        TFAlbertForMultipleChoice,
        TFAlbertForPreTraining,
        TFAlbertForQuestionAnswering,
        TFAlbertForSequenceClassification,
        TFAlbertForTokenClassification,
        TFAlbertMainLayer,
        TFAlbertModel,
        TFAlbertPreTrainedModel,
    )
