__version__ = "1.1.0"
# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
    absl.logging.set_verbosity('info')
    absl.logging.set_stderrthreshold('info')
    absl.logging._warn_preinit_stderr = False
except:
    pass


# Generic
from .configuration_utils import PretrainedConfig, CONFIG_NAME
from .file_utils import (PYTORCH_TRANSFORMERS_CACHE, PYTORCH_PRETRAINED_BERT_CACHE, cached_path,
                         WEIGHTS_NAME, TF_WEIGHTS_NAME)


# Tokenizers
from .tokenization_auto import AutoTokenizer
from .tokenization_bert import BertTokenizer, BasicTokenizer, WordpieceTokenizer
from .tokenization_openai import OpenAIGPTTokenizer
from .tokenization_transfo_xl import (TransfoXLTokenizer, TransfoXLCorpus)
from .tokenization_gpt2 import GPT2Tokenizer
from .tokenization_xlnet import XLNetTokenizer, SPIECE_UNDERLINE
from .tokenization_xlm import XLMTokenizer
from .tokenization_roberta import RobertaTokenizer
from .tokenization_utils import (PreTrainedTokenizer)


# Configurations
from .configuration_auto import AutoConfig
from .configuration_bert import BertConfig, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_openai import OpenAIGPTConfig, OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_transfo_xl import TransfoXLConfig, TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_gpt2 import GPT2Config, GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_xlnet import XLNetConfig, XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_xlm import XLMConfig, XLM_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_roberta import RobertaConfig, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_utils import PretrainedConfig


# PyTorch
from .optimization import (AdamW, ConstantLRSchedule, WarmupConstantSchedule, WarmupCosineSchedule,
                           WarmupCosineWithHardRestartsSchedule, WarmupLinearSchedule)

from .modeling_auto import AutoModel
from .modeling_bert import (BertPreTrainedModel, BertModel, BertForPreTraining,
                            BertForMaskedLM, BertForNextSentencePrediction,
                            BertForSequenceClassification, BertForMultipleChoice,
                            BertForTokenClassification, BertForQuestionAnswering,
                            load_tf_weights_in_bert, BERT_PRETRAINED_MODEL_ARCHIVE_MAP)
from .modeling_openai import (OpenAIGPTPreTrainedModel, OpenAIGPTModel,
                              OpenAIGPTLMHeadModel, OpenAIGPTDoubleHeadsModel,
                              load_tf_weights_in_openai_gpt, OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP)
from .modeling_transfo_xl import (TransfoXLPreTrainedModel, TransfoXLModel, TransfoXLLMHeadModel,
                                  load_tf_weights_in_transfo_xl, TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP)
from .modeling_gpt2 import (GPT2PreTrainedModel, GPT2Model,
                            GPT2LMHeadModel, GPT2DoubleHeadsModel,
                            load_tf_weights_in_gpt2, GPT2_PRETRAINED_MODEL_ARCHIVE_MAP)
from .modeling_xlnet import (XLNetPreTrainedModel, XLNetModel, XLNetLMHeadModel,
                             XLNetForSequenceClassification, XLNetForQuestionAnswering,
                             load_tf_weights_in_xlnet, XLNET_PRETRAINED_MODEL_ARCHIVE_MAP)
from .modeling_xlm import (XLMPreTrainedModel , XLMModel,
                           XLMWithLMHeadModel, XLMForSequenceClassification,
                           XLMForQuestionAnswering, XLM_PRETRAINED_MODEL_ARCHIVE_MAP)
from .modeling_roberta import (RobertaForMaskedLM, RobertaModel, RobertaForSequenceClassification,
                               ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP)
from .modeling_utils import PreTrainedModel, prune_layer, Conv1D


# TensorFlow
from .modeling_tf_utils import TFPreTrainedModel
from .modeling_tf_bert import (TFBertPreTrainedModel, TFBertModel, TFBertForPreTraining,
                               TFBertForMaskedLM, TFBertForNextSentencePrediction, load_pt_weights_in_bert)
