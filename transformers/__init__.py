__version__ = "2.2.1"

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

# Files and general utilities
from .file_utils import (TRANSFORMERS_CACHE, PYTORCH_TRANSFORMERS_CACHE, PYTORCH_PRETRAINED_BERT_CACHE,
                         cached_path, add_start_docstrings, add_end_docstrings,
                         WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, CONFIG_NAME,
                         is_tf_available, is_torch_available)

from .data import (is_sklearn_available,
                   InputExample, InputFeatures, DataProcessor,
                   glue_output_modes, glue_convert_examples_to_features,
                   glue_processors, glue_tasks_num_labels,
                   xnli_output_modes, xnli_processors, xnli_tasks_num_labels,
                   squad_convert_examples_to_features, SquadFeatures, 
                   SquadExample, SquadV1Processor, SquadV2Processor)

if is_sklearn_available():
    from .data import glue_compute_metrics, xnli_compute_metrics

# Tokenizers
from .tokenization_utils import (PreTrainedTokenizer)
from .tokenization_auto import AutoTokenizer
from .tokenization_bert import BertTokenizer, BasicTokenizer, WordpieceTokenizer
from .tokenization_bert_japanese import BertJapaneseTokenizer, MecabTokenizer, CharacterTokenizer
from .tokenization_openai import OpenAIGPTTokenizer
from .tokenization_transfo_xl import (TransfoXLTokenizer, TransfoXLCorpus)
from .tokenization_gpt2 import GPT2Tokenizer
from .tokenization_ctrl import CTRLTokenizer
from .tokenization_xlnet import XLNetTokenizer, SPIECE_UNDERLINE
from .tokenization_xlm import XLMTokenizer
from .tokenization_roberta import RobertaTokenizer
from .tokenization_distilbert import DistilBertTokenizer
from .tokenization_albert import AlbertTokenizer
from .tokenization_camembert import CamembertTokenizer

# Configurations
from .configuration_utils import PretrainedConfig
from .configuration_auto import AutoConfig
from .configuration_bert import BertConfig, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_openai import OpenAIGPTConfig, OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_transfo_xl import TransfoXLConfig, TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_gpt2 import GPT2Config, GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_ctrl import CTRLConfig, CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_xlnet import XLNetConfig, XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_ctrl import CTRLConfig, CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_xlm import XLMConfig, XLM_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_roberta import RobertaConfig, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_distilbert import DistilBertConfig, DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_albert import AlbertConfig, ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_camembert import CamembertConfig, CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP

# Modeling
if is_torch_available():
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
                                    AdaptiveEmbedding,
                                    load_tf_weights_in_transfo_xl, TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_gpt2 import (GPT2PreTrainedModel, GPT2Model,
                                GPT2LMHeadModel, GPT2DoubleHeadsModel,
                                load_tf_weights_in_gpt2, GPT2_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_ctrl import (CTRLPreTrainedModel, CTRLModel,
                                CTRLLMHeadModel,
                                CTRL_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_xlnet import (XLNetPreTrainedModel, XLNetModel, XLNetLMHeadModel,
                                XLNetForSequenceClassification, XLNetForTokenClassification,
                                XLNetForMultipleChoice, XLNetForQuestionAnsweringSimple,
                                XLNetForQuestionAnswering, load_tf_weights_in_xlnet,
                                XLNET_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_xlm import (XLMPreTrainedModel , XLMModel,
                            XLMWithLMHeadModel, XLMForSequenceClassification,
                            XLMForQuestionAnswering, XLMForQuestionAnsweringSimple,
                            XLM_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_roberta import (RobertaForMaskedLM, RobertaModel,
                                RobertaForSequenceClassification, RobertaForMultipleChoice,
                                RobertaForTokenClassification,
                                ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_distilbert import (DistilBertPreTrainedModel, DistilBertForMaskedLM, DistilBertModel,
                                DistilBertForSequenceClassification, DistilBertForQuestionAnswering,
                                DistilBertForTokenClassification,
                                DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_camembert import (CamembertForMaskedLM, CamembertModel,
                                CamembertForSequenceClassification, CamembertForMultipleChoice,
                                CamembertForTokenClassification,
                                CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_encoder_decoder import PreTrainedEncoderDecoder, Model2Model

    from .modeling_albert import (AlbertPreTrainedModel, AlbertModel, AlbertForMaskedLM, AlbertForSequenceClassification,
                                AlbertForQuestionAnswering,
                                load_tf_weights_in_albert, ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP)

    # Optimization
    from .optimization import (AdamW, get_constant_schedule, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup,
                               get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup)


# TensorFlow
if is_tf_available():
    from .modeling_tf_utils import TFPreTrainedModel, TFSharedEmbeddings, TFSequenceSummary, shape_list
    from .modeling_tf_auto import (TFAutoModel, TFAutoModelForSequenceClassification, TFAutoModelForQuestionAnswering,
                                   TFAutoModelWithLMHead)

    from .modeling_tf_bert import (TFBertPreTrainedModel, TFBertMainLayer, TFBertEmbeddings,
                                   TFBertModel, TFBertForPreTraining,
                                   TFBertForMaskedLM, TFBertForNextSentencePrediction,
                                   TFBertForSequenceClassification, TFBertForMultipleChoice,
                                   TFBertForTokenClassification, TFBertForQuestionAnswering,
                                   TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP)

    from .modeling_tf_gpt2 import (TFGPT2PreTrainedModel, TFGPT2MainLayer,
                                   TFGPT2Model, TFGPT2LMHeadModel, TFGPT2DoubleHeadsModel,
                                   TF_GPT2_PRETRAINED_MODEL_ARCHIVE_MAP)

    from .modeling_tf_openai import (TFOpenAIGPTPreTrainedModel, TFOpenAIGPTMainLayer,
                                     TFOpenAIGPTModel, TFOpenAIGPTLMHeadModel, TFOpenAIGPTDoubleHeadsModel,
                                     TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP)

    from .modeling_tf_transfo_xl import (TFTransfoXLPreTrainedModel, TFTransfoXLMainLayer,
                                         TFTransfoXLModel, TFTransfoXLLMHeadModel,
                                         TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP)

    from .modeling_tf_xlnet import (TFXLNetPreTrainedModel, TFXLNetMainLayer,
                                    TFXLNetModel, TFXLNetLMHeadModel,
                                    TFXLNetForSequenceClassification,
                                    TFXLNetForTokenClassification,
                                    TFXLNetForQuestionAnsweringSimple,
                                    TF_XLNET_PRETRAINED_MODEL_ARCHIVE_MAP)

    from .modeling_tf_xlm import (TFXLMPreTrainedModel, TFXLMMainLayer,
                                  TFXLMModel, TFXLMWithLMHeadModel,
                                  TFXLMForSequenceClassification,
                                  TFXLMForQuestionAnsweringSimple,
                                  TF_XLM_PRETRAINED_MODEL_ARCHIVE_MAP)

    from .modeling_tf_roberta import (TFRobertaPreTrainedModel, TFRobertaMainLayer,
                                      TFRobertaModel, TFRobertaForMaskedLM,
                                      TFRobertaForSequenceClassification,
                                      TFRobertaForTokenClassification,
                                      TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP)

    from .modeling_tf_distilbert import (TFDistilBertPreTrainedModel, TFDistilBertMainLayer,
                                         TFDistilBertModel, TFDistilBertForMaskedLM,
                                         TFDistilBertForSequenceClassification,
                                         TFDistilBertForTokenClassification,
                                         TFDistilBertForQuestionAnswering,
                                         TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP)

    from .modeling_tf_ctrl import (TFCTRLPreTrainedModel, TFCTRLModel,
                                    TFCTRLLMHeadModel,
                                    TF_CTRL_PRETRAINED_MODEL_ARCHIVE_MAP)

    from .modeling_tf_albert import (TFAlbertPreTrainedModel, TFAlbertModel, TFAlbertForMaskedLM,
                                     TFAlbertForSequenceClassification,
                                    TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP)
    # Optimization
    from .optimization_tf import (WarmUp, create_optimizer, AdamWeightDecay, GradientAccumulator)

# TF 2.0 <=> PyTorch conversion utilities
from .modeling_tf_pytorch_utils import (convert_tf_weight_name_to_pt_weight_name,
                                        load_pytorch_checkpoint_in_tf2_model,
                                        load_pytorch_weights_in_tf2_model,
                                        load_pytorch_model_in_tf2_model,
                                        load_tf2_checkpoint_in_pytorch_model,
                                        load_tf2_weights_in_pytorch_model,
                                        load_tf2_model_in_pytorch_model)

if not is_tf_available() and not is_torch_available():
    logger.warning("Neither PyTorch nor TensorFlow >= 2.0 have been found."
                   "Models won't be available and only tokenizers, configuration"
                   "and file/data utilities can be used.")
