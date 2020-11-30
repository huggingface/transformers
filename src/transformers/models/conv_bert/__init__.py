# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.
from ...file_utils import is_torch_available
from .configuration_conv_bert import CONV_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, ConvBertConfig
from .tokenization_conv_bert import ConvBertTokenizer


if is_torch_available():
    from .modeling_conv_bert import (
        CONV_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        ConvBertForMaskedLM,
        ConvBertForMultipleChoice,
        ConvBertForQuestionAnswering,
        ConvBertForSequenceClassification,
        ConvBertForTokenClassification,
        ConvBertLayer,
        ConvBertModel,
        ConvBertPreTrainedModel,
        load_tf_weights_in_conv_bert,
    )
