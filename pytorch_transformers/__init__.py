__version__ = "1.2.0"

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

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# Tokenizer
from .tokenization_utils import (PreTrainedTokenizer)
from .tokenization_auto import AutoTokenizer
from .tokenization_bert import BertTokenizer, BasicTokenizer, WordpieceTokenizer
from .tokenization_openai import OpenAIGPTTokenizer
from .tokenization_transfo_xl import (TransfoXLTokenizer, TransfoXLCorpus)
from .tokenization_gpt2 import GPT2Tokenizer
from .tokenization_xlnet import XLNetTokenizer, SPIECE_UNDERLINE
from .tokenization_xlm import XLMTokenizer
from .tokenization_roberta import RobertaTokenizer
from .tokenization_distilbert import DistilBertTokenizer

# Configurations
from .configuration_utils import PretrainedConfig
from .configuration_auto import AutoConfig
from .configuration_bert import BertConfig, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_openai import OpenAIGPTConfig, OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_transfo_xl import TransfoXLConfig, TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_gpt2 import GPT2Config, GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_xlnet import XLNetConfig, XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_xlm import XLMConfig, XLM_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_roberta import RobertaConfig, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_distilbert import DistilBertConfig, DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP

# Modeling
try:
    import torch
    _torch_available = True  # pylint: disable=invalid-name
except ImportError:
    _torch_available = False  # pylint: disable=invalid-name

if _torch_available:
    logger.info("PyTorch version {} available.".format(torch.__version__))

    from .modeling_utils import (PreTrainedModel, prune_layer, Conv1D)
    from .modeling_auto import (AutoModel, AutoModelForSequenceClassification, AutoModelForQuestionAnswering,
                                AutoModelWithLMHead)

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
                                XLNetForSequenceClassification, XLNetForQuestionAnsweringSimple,
                                XLNetForQuestionAnswering,
                                load_tf_weights_in_xlnet, XLNET_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_xlm import (XLMPreTrainedModel , XLMModel,
                            XLMWithLMHeadModel, XLMForSequenceClassification,
                            XLMForQuestionAnswering, XLM_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_roberta import (RobertaForMaskedLM, RobertaModel, RobertaForSequenceClassification,
                                ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_distilbert import (DistilBertForMaskedLM, DistilBertModel,
                                DistilBertForSequenceClassification, DistilBertForQuestionAnswering,
                                DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP)

    # Optimization
    from .optimization import (AdamW, ConstantLRSchedule, WarmupConstantSchedule, WarmupCosineSchedule,
                               WarmupCosineWithHardRestartsSchedule, WarmupLinearSchedule)


# TensorFlow
try:
    import tensorflow as tf
    assert int(tf.__version__[0]) >= 2
    _tf_available = True  # pylint: disable=invalid-name
except (ImportError, AssertionError):
    _tf_available = False  # pylint: disable=invalid-name

if _tf_available:
    logger.info("TensorFlow version {} available.".format(tf.__version__))

    from .modeling_tf_utils import TFPreTrainedModel, TFSharedEmbeddings, TFSequenceSummary
    from .modeling_tf_auto import (TFAutoModel, TFAutoModelForSequenceClassification, TFAutoModelForQuestionAnswering,
                                   TFAutoModelWithLMHead)

    from .modeling_tf_bert import (TFBertPreTrainedModel, TFBertMainLayer, TFBertEmbeddings,
                                   TFBertModel, TFBertForPreTraining,
                                   TFBertForMaskedLM, TFBertForNextSentencePrediction,
                                   TFBertForSequenceClassification, TFBertForMultipleChoice,
                                   TFBertForTokenClassification, TFBertForQuestionAnswering,
                                   load_bert_pt_weights_in_tf2,
                                   TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP)

    from .modeling_tf_gpt2 import (TFGPT2PreTrainedModel, TFGPT2MainLayer,
                                   TFGPT2Model, TFGPT2LMHeadModel, TFGPT2DoubleHeadsModel,
                                   load_gpt2_pt_weights_in_tf2,
                                   TF_GPT2_PRETRAINED_MODEL_ARCHIVE_MAP)

    from .modeling_tf_transfo_xl import (TFTransfoXLPreTrainedModel, TFTransfoXLMainLayer,
                                         TFTransfoXLModel, TFTransfoXLLMHeadModel,
                                         load_transfo_xl_pt_weights_in_tf2,
                                         TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP)

    from .modeling_tf_xlnet import (TFXLNetPreTrainedModel, TFXLNetMainLayer,
                                    TFXLNetModel, TFXLNetLMHeadModel,
                                    TFXLNetForSequenceClassification,
                                    TFXLNetForQuestionAnsweringSimple,
                                    load_xlnet_pt_weights_in_tf2,
                                    TF_XLNET_PRETRAINED_MODEL_ARCHIVE_MAP)

    from .modeling_tf_xlm import (TFXLMPreTrainedModel, TFXLMMainLayer,
                                  TFXLMModel, TFXLMWithLMHeadModel,
                                  TFXLMForSequenceClassification,
                                  TFXLMForQuestionAnsweringSimple,
                                  load_xlm_pt_weights_in_tf2,
                                  TF_XLM_PRETRAINED_MODEL_ARCHIVE_MAP)

# Files and general utilities
from .file_utils import (PYTORCH_TRANSFORMERS_CACHE, PYTORCH_PRETRAINED_BERT_CACHE,
                         cached_path, add_start_docstrings, add_end_docstrings,
                         WEIGHTS_NAME, TF_WEIGHTS_NAME, CONFIG_NAME)

def is_torch_available():
    return _torch_available

def is_tf_available():
    return _tf_available
