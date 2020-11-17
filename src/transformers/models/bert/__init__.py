# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_flax_available, is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
from .tokenization_bert import BasicTokenizer, BertTokenizer, WordpieceTokenizer


if is_tokenizers_available():
    from .tokenization_bert_fast import BertTokenizerFast

if is_torch_available():
    from .modeling_bert import (
        BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        BertForMaskedLM,
        BertForMultipleChoice,
        BertForNextSentencePrediction,
        BertForPreTraining,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        BertForTokenClassification,
        BertLayer,
        BertLMHeadModel,
        BertModel,
        BertPreTrainedModel,
        load_tf_weights_in_bert,
    )

if is_tf_available():
    from .modeling_tf_bert import (
        TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFBertEmbeddings,
        TFBertForMaskedLM,
        TFBertForMultipleChoice,
        TFBertForNextSentencePrediction,
        TFBertForPreTraining,
        TFBertForQuestionAnswering,
        TFBertForSequenceClassification,
        TFBertForTokenClassification,
        TFBertLMHeadModel,
        TFBertMainLayer,
        TFBertModel,
        TFBertPreTrainedModel,
    )

if is_flax_available():
    from .modeling_flax_bert import FlaxBertModel
