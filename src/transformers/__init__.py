# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__version__ = "4.1.0.dev0"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

from . import dependency_versions_check

# Configuration
from .configuration_utils import PretrainedConfig

# Data
from .data import (
    DataProcessor,
    InputExample,
    InputFeatures,
    SingleSentenceClassificationProcessor,
    SquadExample,
    SquadFeatures,
    SquadV1Processor,
    SquadV2Processor,
    glue_compute_metrics,
    glue_convert_examples_to_features,
    glue_output_modes,
    glue_processors,
    glue_tasks_num_labels,
    squad_convert_examples_to_features,
    xnli_compute_metrics,
    xnli_output_modes,
    xnli_processors,
    xnli_tasks_num_labels,
)

# Files and general utilities
from .file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    SPIECE_UNDERLINE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
    is_apex_available,
    is_datasets_available,
    is_faiss_available,
    is_flax_available,
    is_psutil_available,
    is_py3nvml_available,
    is_sentencepiece_available,
    is_sklearn_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    is_torch_tpu_available,
)
from .hf_argparser import HfArgumentParser

# Model Cards
from .modelcard import ModelCard

# TF 2.0 <=> PyTorch conversion utilities
from .modeling_tf_pytorch_utils import (
    convert_tf_weight_name_to_pt_weight_name,
    load_pytorch_checkpoint_in_tf2_model,
    load_pytorch_model_in_tf2_model,
    load_pytorch_weights_in_tf2_model,
    load_tf2_checkpoint_in_pytorch_model,
    load_tf2_model_in_pytorch_model,
    load_tf2_weights_in_pytorch_model,
)
from .models.albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
from .models.auto import (
    ALL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    CONFIG_MAPPING,
    MODEL_NAMES_MAPPING,
    TOKENIZER_MAPPING,
    AutoConfig,
    AutoTokenizer,
)
from .models.bart import BartConfig, BartTokenizer
from .models.bert import (
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BasicTokenizer,
    BertConfig,
    BertTokenizer,
    WordpieceTokenizer,
)
from .models.bert_generation import BertGenerationConfig
from .models.bert_japanese import BertJapaneseTokenizer, CharacterTokenizer, MecabTokenizer
from .models.bertweet import BertweetTokenizer
from .models.blenderbot import (
    BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BlenderbotConfig,
    BlenderbotSmallTokenizer,
    BlenderbotTokenizer,
)
from .models.camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CamembertConfig
from .models.ctrl import CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRLConfig, CTRLTokenizer
from .models.deberta import DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, DebertaConfig, DebertaTokenizer
from .models.distilbert import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DistilBertConfig, DistilBertTokenizer
from .models.dpr import (
    DPR_PRETRAINED_CONFIG_ARCHIVE_MAP,
    DPRConfig,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoderTokenizer,
    DPRReaderOutput,
    DPRReaderTokenizer,
)
from .models.electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig, ElectraTokenizer
from .models.encoder_decoder import EncoderDecoderConfig
from .models.flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig, FlaubertTokenizer
from .models.fsmt import FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP, FSMTConfig, FSMTTokenizer
from .models.funnel import FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP, FunnelConfig, FunnelTokenizer
from .models.gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config, GPT2Tokenizer
from .models.herbert import HerbertTokenizer
from .models.layoutlm import LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP, LayoutLMConfig, LayoutLMTokenizer
from .models.longformer import LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, LongformerConfig, LongformerTokenizer
from .models.lxmert import LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP, LxmertConfig, LxmertTokenizer
from .models.marian import MarianConfig
from .models.mbart import MBartConfig
from .models.mmbt import MMBTConfig
from .models.mobilebert import MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, MobileBertConfig, MobileBertTokenizer
from .models.mpnet import MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP, MPNetConfig, MPNetTokenizer
from .models.mt5 import MT5Config
from .models.openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig, OpenAIGPTTokenizer
from .models.pegasus import PegasusConfig
from .models.phobert import PhobertTokenizer
from .models.prophetnet import PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ProphetNetConfig, ProphetNetTokenizer
from .models.rag import RagConfig, RagRetriever, RagTokenizer
from .models.reformer import REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, ReformerConfig
from .models.retribert import RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RetriBertConfig, RetriBertTokenizer
from .models.roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig, RobertaTokenizer
from .models.squeezebert import SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, SqueezeBertConfig, SqueezeBertTokenizer
from .models.t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config
from .models.transfo_xl import (
    TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    TransfoXLConfig,
    TransfoXLCorpus,
    TransfoXLTokenizer,
)
from .models.xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig, XLMTokenizer
from .models.xlm_prophetnet import XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMProphetNetConfig
from .models.xlm_roberta import XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMRobertaConfig
from .models.xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig

# Pipelines
from .pipelines import (
    Conversation,
    ConversationalPipeline,
    CsvPipelineDataFormat,
    FeatureExtractionPipeline,
    FillMaskPipeline,
    JsonPipelineDataFormat,
    NerPipeline,
    PipedPipelineDataFormat,
    Pipeline,
    PipelineDataFormat,
    QuestionAnsweringPipeline,
    SummarizationPipeline,
    Text2TextGenerationPipeline,
    TextClassificationPipeline,
    TextGenerationPipeline,
    TokenClassificationPipeline,
    TranslationPipeline,
    ZeroShotClassificationPipeline,
    pipeline,
)

# Tokenization
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import (
    AddedToken,
    BatchEncoding,
    CharSpan,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TensorType,
    TokenSpan,
)


# Integrations: this needs to come before other ml imports
# in order to allow any 3rd-party code to initialize properly
from .integrations import (  # isort:skip
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)


if is_sentencepiece_available():
    from .models.albert import AlbertTokenizer
    from .models.barthez import BarthezTokenizer
    from .models.bert_generation import BertGenerationTokenizer
    from .models.camembert import CamembertTokenizer
    from .models.marian import MarianTokenizer
    from .models.mbart import MBartTokenizer
    from .models.mt5 import MT5Tokenizer
    from .models.pegasus import PegasusTokenizer
    from .models.reformer import ReformerTokenizer
    from .models.t5 import T5Tokenizer
    from .models.xlm_prophetnet import XLMProphetNetTokenizer
    from .models.xlm_roberta import XLMRobertaTokenizer
    from .models.xlnet import XLNetTokenizer
else:
    from .utils.dummy_sentencepiece_objects import *

if is_tokenizers_available():
    from .models.albert import AlbertTokenizerFast
    from .models.bart import BartTokenizerFast
    from .models.barthez import BarthezTokenizerFast
    from .models.bert import BertTokenizerFast
    from .models.camembert import CamembertTokenizerFast
    from .models.distilbert import DistilBertTokenizerFast
    from .models.dpr import DPRContextEncoderTokenizerFast, DPRQuestionEncoderTokenizerFast, DPRReaderTokenizerFast
    from .models.electra import ElectraTokenizerFast
    from .models.funnel import FunnelTokenizerFast
    from .models.gpt2 import GPT2TokenizerFast
    from .models.herbert import HerbertTokenizerFast
    from .models.layoutlm import LayoutLMTokenizerFast
    from .models.longformer import LongformerTokenizerFast
    from .models.lxmert import LxmertTokenizerFast
    from .models.mbart import MBartTokenizerFast
    from .models.mobilebert import MobileBertTokenizerFast
    from .models.mpnet import MPNetTokenizerFast
    from .models.mt5 import MT5TokenizerFast
    from .models.openai import OpenAIGPTTokenizerFast
    from .models.pegasus import PegasusTokenizerFast
    from .models.reformer import ReformerTokenizerFast
    from .models.retribert import RetriBertTokenizerFast
    from .models.roberta import RobertaTokenizerFast
    from .models.squeezebert import SqueezeBertTokenizerFast
    from .models.t5 import T5TokenizerFast
    from .models.xlm_roberta import XLMRobertaTokenizerFast
    from .models.xlnet import XLNetTokenizerFast
    from .tokenization_utils_fast import PreTrainedTokenizerFast

    if is_sentencepiece_available():
        from .convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, convert_slow_tokenizer
else:
    from .utils.dummy_tokenizers_objects import *

# Trainer
from .trainer_callback import (
    DefaultFlowCallback,
    EarlyStoppingCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer_utils import EvalPrediction, EvaluationStrategy, set_seed
from .training_args import TrainingArguments
from .training_args_tf import TFTrainingArguments
from .utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Modeling
if is_torch_available():
    # Benchmarks
    from .benchmark.benchmark import PyTorchBenchmark
    from .benchmark.benchmark_args import PyTorchBenchmarkArguments
    from .data.data_collator import (
        DataCollator,
        DataCollatorForLanguageModeling,
        DataCollatorForPermutationLanguageModeling,
        DataCollatorForSOP,
        DataCollatorForTokenClassification,
        DataCollatorForWholeWordMask,
        DataCollatorWithPadding,
        default_data_collator,
    )
    from .data.datasets import (
        GlueDataset,
        GlueDataTrainingArguments,
        LineByLineTextDataset,
        LineByLineWithRefDataset,
        LineByLineWithSOPTextDataset,
        SquadDataset,
        SquadDataTrainingArguments,
        TextDataset,
        TextDatasetForNextSentencePrediction,
    )
    from .generation_beam_search import BeamScorer, BeamSearchScorer
    from .generation_logits_process import (
        HammingDiversityLogitsProcessor,
        LogitsProcessor,
        LogitsProcessorList,
        LogitsWarper,
        MinLengthLogitsProcessor,
        NoBadWordsLogitsProcessor,
        NoRepeatNGramLogitsProcessor,
        PrefixConstrainedLogitsProcessor,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )
    from .generation_utils import top_k_top_p_filtering
    from .modeling_utils import Conv1D, PreTrainedModel, apply_chunking_to_forward, prune_layer
    from .models.albert import (
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
    from .models.auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING,
        MODEL_FOR_MASKED_LM_MAPPING,
        MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
        MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
        MODEL_FOR_PRETRAINING_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        MODEL_MAPPING,
        MODEL_WITH_LM_HEAD_MAPPING,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForMultipleChoice,
        AutoModelForNextSentencePrediction,
        AutoModelForPreTraining,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoModelWithLMHead,
    )
    from .models.bart import (
        BART_PRETRAINED_MODEL_ARCHIVE_LIST,
        BartForConditionalGeneration,
        BartForQuestionAnswering,
        BartForSequenceClassification,
        BartModel,
        BartPretrainedModel,
        PretrainedBartModel,
    )
    from .models.bert import (
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
    from .models.bert_generation import (
        BertGenerationDecoder,
        BertGenerationEncoder,
        load_tf_weights_in_bert_generation,
    )
    from .models.blenderbot import BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST, BlenderbotForConditionalGeneration
    from .models.camembert import (
        CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        CamembertForCausalLM,
        CamembertForMaskedLM,
        CamembertForMultipleChoice,
        CamembertForQuestionAnswering,
        CamembertForSequenceClassification,
        CamembertForTokenClassification,
        CamembertModel,
    )
    from .models.ctrl import (
        CTRL_PRETRAINED_MODEL_ARCHIVE_LIST,
        CTRLForSequenceClassification,
        CTRLLMHeadModel,
        CTRLModel,
        CTRLPreTrainedModel,
    )
    from .models.deberta import (
        DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        DebertaForSequenceClassification,
        DebertaModel,
        DebertaPreTrainedModel,
    )
    from .models.distilbert import (
        DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        DistilBertForMaskedLM,
        DistilBertForMultipleChoice,
        DistilBertForQuestionAnswering,
        DistilBertForSequenceClassification,
        DistilBertForTokenClassification,
        DistilBertModel,
        DistilBertPreTrainedModel,
    )
    from .models.dpr import (
        DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST,
        DPRContextEncoder,
        DPRPretrainedContextEncoder,
        DPRPretrainedQuestionEncoder,
        DPRPretrainedReader,
        DPRQuestionEncoder,
        DPRReader,
    )
    from .models.electra import (
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
    from .models.encoder_decoder import EncoderDecoderModel
    from .models.flaubert import (
        FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        FlaubertForMultipleChoice,
        FlaubertForQuestionAnswering,
        FlaubertForQuestionAnsweringSimple,
        FlaubertForSequenceClassification,
        FlaubertForTokenClassification,
        FlaubertModel,
        FlaubertWithLMHeadModel,
    )
    from .models.fsmt import FSMTForConditionalGeneration, FSMTModel, PretrainedFSMTModel
    from .models.funnel import (
        FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST,
        FunnelBaseModel,
        FunnelForMaskedLM,
        FunnelForMultipleChoice,
        FunnelForPreTraining,
        FunnelForQuestionAnswering,
        FunnelForSequenceClassification,
        FunnelForTokenClassification,
        FunnelModel,
        load_tf_weights_in_funnel,
    )
    from .models.gpt2 import (
        GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
        GPT2DoubleHeadsModel,
        GPT2ForSequenceClassification,
        GPT2LMHeadModel,
        GPT2Model,
        GPT2PreTrainedModel,
        load_tf_weights_in_gpt2,
    )
    from .models.layoutlm import (
        LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,
        LayoutLMForMaskedLM,
        LayoutLMForTokenClassification,
        LayoutLMModel,
    )
    from .models.longformer import (
        LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
        LongformerForMaskedLM,
        LongformerForMultipleChoice,
        LongformerForQuestionAnswering,
        LongformerForSequenceClassification,
        LongformerForTokenClassification,
        LongformerModel,
        LongformerSelfAttention,
    )
    from .models.lxmert import (
        LxmertEncoder,
        LxmertForPreTraining,
        LxmertForQuestionAnswering,
        LxmertModel,
        LxmertPreTrainedModel,
        LxmertVisualFeatureEncoder,
        LxmertXLayer,
    )
    from .models.marian import MarianMTModel
    from .models.mbart import MBartForConditionalGeneration
    from .models.mmbt import MMBTForClassification, MMBTModel, ModalEmbeddings
    from .models.mobilebert import (
        MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        MobileBertForMaskedLM,
        MobileBertForMultipleChoice,
        MobileBertForNextSentencePrediction,
        MobileBertForPreTraining,
        MobileBertForQuestionAnswering,
        MobileBertForSequenceClassification,
        MobileBertForTokenClassification,
        MobileBertLayer,
        MobileBertModel,
        MobileBertPreTrainedModel,
        load_tf_weights_in_mobilebert,
    )
    from .models.mpnet import (
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
    from .models.mt5 import MT5EncoderModel, MT5ForConditionalGeneration, MT5Model
    from .models.openai import (
        OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,
        OpenAIGPTDoubleHeadsModel,
        OpenAIGPTForSequenceClassification,
        OpenAIGPTLMHeadModel,
        OpenAIGPTModel,
        OpenAIGPTPreTrainedModel,
        load_tf_weights_in_openai_gpt,
    )
    from .models.pegasus import PegasusForConditionalGeneration
    from .models.prophetnet import (
        PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST,
        ProphetNetDecoder,
        ProphetNetEncoder,
        ProphetNetForCausalLM,
        ProphetNetForConditionalGeneration,
        ProphetNetModel,
        ProphetNetPreTrainedModel,
    )
    from .models.rag import RagModel, RagSequenceForGeneration, RagTokenForGeneration
    from .models.reformer import (
        REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
        ReformerAttention,
        ReformerForMaskedLM,
        ReformerForQuestionAnswering,
        ReformerForSequenceClassification,
        ReformerLayer,
        ReformerModel,
        ReformerModelWithLMHead,
    )
    from .models.retribert import RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST, RetriBertModel, RetriBertPreTrainedModel
    from .models.roberta import (
        ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        RobertaForCausalLM,
        RobertaForMaskedLM,
        RobertaForMultipleChoice,
        RobertaForQuestionAnswering,
        RobertaForSequenceClassification,
        RobertaForTokenClassification,
        RobertaModel,
    )
    from .models.squeezebert import (
        SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        SqueezeBertForMaskedLM,
        SqueezeBertForMultipleChoice,
        SqueezeBertForQuestionAnswering,
        SqueezeBertForSequenceClassification,
        SqueezeBertForTokenClassification,
        SqueezeBertModel,
        SqueezeBertModule,
        SqueezeBertPreTrainedModel,
    )
    from .models.t5 import (
        T5_PRETRAINED_MODEL_ARCHIVE_LIST,
        T5EncoderModel,
        T5ForConditionalGeneration,
        T5Model,
        T5PreTrainedModel,
        load_tf_weights_in_t5,
    )
    from .models.transfo_xl import (
        TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
        AdaptiveEmbedding,
        TransfoXLForSequenceClassification,
        TransfoXLLMHeadModel,
        TransfoXLModel,
        TransfoXLPreTrainedModel,
        load_tf_weights_in_transfo_xl,
    )
    from .models.xlm import (
        XLM_PRETRAINED_MODEL_ARCHIVE_LIST,
        XLMForMultipleChoice,
        XLMForQuestionAnswering,
        XLMForQuestionAnsweringSimple,
        XLMForSequenceClassification,
        XLMForTokenClassification,
        XLMModel,
        XLMPreTrainedModel,
        XLMWithLMHeadModel,
    )
    from .models.xlm_prophetnet import (
        XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST,
        XLMProphetNetDecoder,
        XLMProphetNetEncoder,
        XLMProphetNetForCausalLM,
        XLMProphetNetForConditionalGeneration,
        XLMProphetNetModel,
    )
    from .models.xlm_roberta import (
        XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        XLMRobertaForCausalLM,
        XLMRobertaForMaskedLM,
        XLMRobertaForMultipleChoice,
        XLMRobertaForQuestionAnswering,
        XLMRobertaForSequenceClassification,
        XLMRobertaForTokenClassification,
        XLMRobertaModel,
    )
    from .models.xlnet import (
        XLNET_PRETRAINED_MODEL_ARCHIVE_LIST,
        XLNetForMultipleChoice,
        XLNetForQuestionAnswering,
        XLNetForQuestionAnsweringSimple,
        XLNetForSequenceClassification,
        XLNetForTokenClassification,
        XLNetLMHeadModel,
        XLNetModel,
        XLNetPreTrainedModel,
        load_tf_weights_in_xlnet,
    )

    # Optimization
    from .optimization import (
        Adafactor,
        AdamW,
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
    )

    # Trainer
    from .trainer import Trainer
    from .trainer_pt_utils import torch_distributed_zero_first
else:
    from .utils.dummy_pt_objects import *

# TensorFlow
if is_tf_available():
    from .benchmark.benchmark_args_tf import TensorFlowBenchmarkArguments

    # Benchmarks
    from .benchmark.benchmark_tf import TensorFlowBenchmark
    from .generation_tf_utils import tf_top_k_top_p_filtering
    from .modeling_tf_utils import TFPreTrainedModel, TFSequenceSummary, TFSharedEmbeddings, shape_list
    from .models.albert import (
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
    from .models.auto import (
        TF_MODEL_FOR_CAUSAL_LM_MAPPING,
        TF_MODEL_FOR_MASKED_LM_MAPPING,
        TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
        TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
        TF_MODEL_FOR_PRETRAINING_MAPPING,
        TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        TF_MODEL_MAPPING,
        TF_MODEL_WITH_LM_HEAD_MAPPING,
        TFAutoModel,
        TFAutoModelForCausalLM,
        TFAutoModelForMaskedLM,
        TFAutoModelForMultipleChoice,
        TFAutoModelForPreTraining,
        TFAutoModelForQuestionAnswering,
        TFAutoModelForSeq2SeqLM,
        TFAutoModelForSequenceClassification,
        TFAutoModelForTokenClassification,
        TFAutoModelWithLMHead,
    )
    from .models.bart import TFBartForConditionalGeneration, TFBartModel
    from .models.bert import (
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
    from .models.blenderbot import TFBlenderbotForConditionalGeneration
    from .models.camembert import (
        TF_CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFCamembertForMaskedLM,
        TFCamembertForMultipleChoice,
        TFCamembertForQuestionAnswering,
        TFCamembertForSequenceClassification,
        TFCamembertForTokenClassification,
        TFCamembertModel,
    )
    from .models.ctrl import (
        TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFCTRLLMHeadModel,
        TFCTRLModel,
        TFCTRLPreTrainedModel,
    )
    from .models.distilbert import (
        TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFDistilBertForMaskedLM,
        TFDistilBertForMultipleChoice,
        TFDistilBertForQuestionAnswering,
        TFDistilBertForSequenceClassification,
        TFDistilBertForTokenClassification,
        TFDistilBertMainLayer,
        TFDistilBertModel,
        TFDistilBertPreTrainedModel,
    )
    from .models.dpr import (
        TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        TF_DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFDPRContextEncoder,
        TFDPRPretrainedContextEncoder,
        TFDPRPretrainedQuestionEncoder,
        TFDPRPretrainedReader,
        TFDPRQuestionEncoder,
        TFDPRReader,
    )
    from .models.electra import (
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
    from .models.flaubert import (
        TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFFlaubertForMultipleChoice,
        TFFlaubertForQuestionAnsweringSimple,
        TFFlaubertForSequenceClassification,
        TFFlaubertForTokenClassification,
        TFFlaubertModel,
        TFFlaubertWithLMHeadModel,
    )
    from .models.funnel import (
        TF_FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFFunnelBaseModel,
        TFFunnelForMaskedLM,
        TFFunnelForMultipleChoice,
        TFFunnelForPreTraining,
        TFFunnelForQuestionAnswering,
        TFFunnelForSequenceClassification,
        TFFunnelForTokenClassification,
        TFFunnelModel,
    )
    from .models.gpt2 import (
        TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFGPT2DoubleHeadsModel,
        TFGPT2ForSequenceClassification,
        TFGPT2LMHeadModel,
        TFGPT2MainLayer,
        TFGPT2Model,
        TFGPT2PreTrainedModel,
    )
    from .models.longformer import (
        TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFLongformerForMaskedLM,
        TFLongformerForMultipleChoice,
        TFLongformerForQuestionAnswering,
        TFLongformerForSequenceClassification,
        TFLongformerForTokenClassification,
        TFLongformerModel,
        TFLongformerSelfAttention,
    )
    from .models.lxmert import (
        TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFLxmertForPreTraining,
        TFLxmertMainLayer,
        TFLxmertModel,
        TFLxmertPreTrainedModel,
        TFLxmertVisualFeatureEncoder,
    )
    from .models.marian import TFMarianMTModel
    from .models.mbart import TFMBartForConditionalGeneration
    from .models.mobilebert import (
        TF_MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFMobileBertForMaskedLM,
        TFMobileBertForMultipleChoice,
        TFMobileBertForNextSentencePrediction,
        TFMobileBertForPreTraining,
        TFMobileBertForQuestionAnswering,
        TFMobileBertForSequenceClassification,
        TFMobileBertForTokenClassification,
        TFMobileBertMainLayer,
        TFMobileBertModel,
        TFMobileBertPreTrainedModel,
    )
    from .models.mpnet import (
        TF_MPNET_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFMPNetForMaskedLM,
        TFMPNetForMultipleChoice,
        TFMPNetForQuestionAnswering,
        TFMPNetForSequenceClassification,
        TFMPNetForTokenClassification,
        TFMPNetMainLayer,
        TFMPNetModel,
        TFMPNetPreTrainedModel,
    )
    from .models.mt5 import TFMT5EncoderModel, TFMT5ForConditionalGeneration, TFMT5Model
    from .models.openai import (
        TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFOpenAIGPTDoubleHeadsModel,
        TFOpenAIGPTLMHeadModel,
        TFOpenAIGPTMainLayer,
        TFOpenAIGPTModel,
        TFOpenAIGPTPreTrainedModel,
    )
    from .models.pegasus import TFPegasusForConditionalGeneration
    from .models.roberta import (
        TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFRobertaForMaskedLM,
        TFRobertaForMultipleChoice,
        TFRobertaForQuestionAnswering,
        TFRobertaForSequenceClassification,
        TFRobertaForTokenClassification,
        TFRobertaMainLayer,
        TFRobertaModel,
        TFRobertaPreTrainedModel,
    )
    from .models.t5 import (
        TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFT5EncoderModel,
        TFT5ForConditionalGeneration,
        TFT5Model,
        TFT5PreTrainedModel,
    )
    from .models.transfo_xl import (
        TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFAdaptiveEmbedding,
        TFTransfoXLLMHeadModel,
        TFTransfoXLMainLayer,
        TFTransfoXLModel,
        TFTransfoXLPreTrainedModel,
    )
    from .models.xlm import (
        TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFXLMForMultipleChoice,
        TFXLMForQuestionAnsweringSimple,
        TFXLMForSequenceClassification,
        TFXLMForTokenClassification,
        TFXLMMainLayer,
        TFXLMModel,
        TFXLMPreTrainedModel,
        TFXLMWithLMHeadModel,
    )
    from .models.xlm_roberta import (
        TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFXLMRobertaForMaskedLM,
        TFXLMRobertaForMultipleChoice,
        TFXLMRobertaForQuestionAnswering,
        TFXLMRobertaForSequenceClassification,
        TFXLMRobertaForTokenClassification,
        TFXLMRobertaModel,
    )
    from .models.xlnet import (
        TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFXLNetForMultipleChoice,
        TFXLNetForQuestionAnsweringSimple,
        TFXLNetForSequenceClassification,
        TFXLNetForTokenClassification,
        TFXLNetLMHeadModel,
        TFXLNetMainLayer,
        TFXLNetModel,
        TFXLNetPreTrainedModel,
    )

    # Optimization
    from .optimization_tf import AdamWeightDecay, GradientAccumulator, WarmUp, create_optimizer

    # Trainer
    from .trainer_tf import TFTrainer

else:
    # Import the same objects as dummies to get them in the namespace.
    # They will raise an import error if the user tries to instantiate / use them.
    from .utils.dummy_tf_objects import *


if is_flax_available():
    from .models.auto import FLAX_MODEL_MAPPING, FlaxAutoModel
    from .models.bert import FlaxBertForMaskedLM, FlaxBertModel
    from .models.roberta import FlaxRobertaModel
else:
    # Import the same objects as dummies to get them in the namespace.
    # They will raise an import error if the user tries to instantiate / use them.
    from .utils.dummy_flax_objects import *


if not is_tf_available() and not is_torch_available() and not is_flax_available():
    logger.warning(
        "None of PyTorch, TensorFlow >= 2.0, or Flax have been found. "
        "Models won't be available and only tokenizers, configuration "
        "and file/data utilities can be used."
    )
