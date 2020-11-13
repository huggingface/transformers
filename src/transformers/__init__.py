# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "3.5.1"

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

# Integrations: this needs to come before other ml imports
# in order to allow any 3rd-party code to initialize properly
from .integrations import (  # isort:skip
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)

# Configurations
from .configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
from .configuration_auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, CONFIG_MAPPING, AutoConfig
from .configuration_bart import BartConfig
from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
from .configuration_bert_generation import BertGenerationConfig
from .configuration_blenderbot import BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP, BlenderbotConfig
from .configuration_camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CamembertConfig
from .configuration_ctrl import CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRLConfig
from .configuration_deberta import DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, DebertaConfig
from .configuration_distilbert import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DistilBertConfig
from .configuration_dpr import DPR_PRETRAINED_CONFIG_ARCHIVE_MAP, DPRConfig
from .configuration_electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig
from .configuration_encoder_decoder import EncoderDecoderConfig
from .configuration_flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig
from .configuration_fsmt import FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP, FSMTConfig
from .configuration_funnel import FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP, FunnelConfig
from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config
from .configuration_layoutlm import LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP, LayoutLMConfig
from .configuration_longformer import LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, LongformerConfig
from .configuration_lxmert import LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP, LxmertConfig
from .configuration_marian import MarianConfig
from .configuration_mbart import MBartConfig
from .configuration_mmbt import MMBTConfig
from .configuration_mobilebert import MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, MobileBertConfig
from .configuration_openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig
from .configuration_pegasus import PegasusConfig
from .configuration_prophetnet import PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ProphetNetConfig
from .configuration_rag import RagConfig
from .configuration_reformer import REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, ReformerConfig
from .configuration_retribert import RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RetriBertConfig
from .configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig
from .configuration_squeezebert import SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, SqueezeBertConfig
from .configuration_t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config
from .configuration_transfo_xl import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TransfoXLConfig
from .configuration_utils import PretrainedConfig
from .configuration_xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig
from .configuration_xlm_prophetnet import XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMProphetNetConfig
from .configuration_xlm_roberta import XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMRobertaConfig
from .configuration_xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig
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

# Retriever
from .retrieval_rag import RagRetriever

# Tokenizers
from .tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
from .tokenization_bart import BartTokenizer
from .tokenization_bert import BasicTokenizer, BertTokenizer, WordpieceTokenizer
from .tokenization_bert_japanese import BertJapaneseTokenizer, CharacterTokenizer, MecabTokenizer
from .tokenization_bertweet import BertweetTokenizer
from .tokenization_blenderbot import BlenderbotSmallTokenizer, BlenderbotTokenizer
from .tokenization_ctrl import CTRLTokenizer
from .tokenization_deberta import DebertaTokenizer
from .tokenization_distilbert import DistilBertTokenizer
from .tokenization_dpr import (
    DPRContextEncoderTokenizer,
    DPRQuestionEncoderTokenizer,
    DPRReaderOutput,
    DPRReaderTokenizer,
)
from .tokenization_electra import ElectraTokenizer
from .tokenization_flaubert import FlaubertTokenizer
from .tokenization_fsmt import FSMTTokenizer
from .tokenization_funnel import FunnelTokenizer
from .tokenization_gpt2 import GPT2Tokenizer
from .tokenization_herbert import HerbertTokenizer
from .tokenization_layoutlm import LayoutLMTokenizer
from .tokenization_longformer import LongformerTokenizer
from .tokenization_lxmert import LxmertTokenizer
from .tokenization_mobilebert import MobileBertTokenizer
from .tokenization_openai import OpenAIGPTTokenizer
from .tokenization_phobert import PhobertTokenizer
from .tokenization_prophetnet import ProphetNetTokenizer
from .tokenization_rag import RagTokenizer
from .tokenization_retribert import RetriBertTokenizer
from .tokenization_roberta import RobertaTokenizer
from .tokenization_squeezebert import SqueezeBertTokenizer
from .tokenization_transfo_xl import TransfoXLCorpus, TransfoXLTokenizer
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
from .tokenization_xlm import XLMTokenizer


if is_sentencepiece_available():
    from .tokenization_albert import AlbertTokenizer
    from .tokenization_bert_generation import BertGenerationTokenizer
    from .tokenization_camembert import CamembertTokenizer
    from .tokenization_marian import MarianTokenizer
    from .tokenization_mbart import MBartTokenizer
    from .tokenization_pegasus import PegasusTokenizer
    from .tokenization_reformer import ReformerTokenizer
    from .tokenization_t5 import T5Tokenizer
    from .tokenization_xlm_prophetnet import XLMProphetNetTokenizer
    from .tokenization_xlm_roberta import XLMRobertaTokenizer
    from .tokenization_xlnet import XLNetTokenizer
else:
    from .utils.dummy_sentencepiece_objects import *

if is_tokenizers_available():
    from .tokenization_albert_fast import AlbertTokenizerFast
    from .tokenization_bart_fast import BartTokenizerFast
    from .tokenization_bert_fast import BertTokenizerFast
    from .tokenization_camembert_fast import CamembertTokenizerFast
    from .tokenization_distilbert_fast import DistilBertTokenizerFast
    from .tokenization_dpr_fast import (
        DPRContextEncoderTokenizerFast,
        DPRQuestionEncoderTokenizerFast,
        DPRReaderTokenizerFast,
    )
    from .tokenization_electra_fast import ElectraTokenizerFast
    from .tokenization_funnel_fast import FunnelTokenizerFast
    from .tokenization_gpt2_fast import GPT2TokenizerFast
    from .tokenization_herbert_fast import HerbertTokenizerFast
    from .tokenization_layoutlm_fast import LayoutLMTokenizerFast
    from .tokenization_longformer_fast import LongformerTokenizerFast
    from .tokenization_lxmert_fast import LxmertTokenizerFast
    from .tokenization_mbart_fast import MBartTokenizerFast
    from .tokenization_mobilebert_fast import MobileBertTokenizerFast
    from .tokenization_openai_fast import OpenAIGPTTokenizerFast
    from .tokenization_pegasus_fast import PegasusTokenizerFast
    from .tokenization_reformer_fast import ReformerTokenizerFast
    from .tokenization_retribert_fast import RetriBertTokenizerFast
    from .tokenization_roberta_fast import RobertaTokenizerFast
    from .tokenization_squeezebert_fast import SqueezeBertTokenizerFast
    from .tokenization_t5_fast import T5TokenizerFast
    from .tokenization_utils_fast import PreTrainedTokenizerFast
    from .tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast
    from .tokenization_xlnet_fast import XLNetTokenizerFast

    if is_sentencepiece_available():
        from .convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, convert_slow_tokenizer
else:
    from .utils.dummy_tokenizers_objects import *

# Trainer
from .trainer_callback import (
    DefaultFlowCallback,
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
        LogitsProcessor,
        LogitsProcessorList,
        LogitsWarper,
        MinLengthLogitsProcessor,
        NoBadWordsLogitsProcessor,
        NoRepeatNGramLogitsProcessor,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )
    from .generation_utils import top_k_top_p_filtering
    from .modeling_albert import (
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
    from .modeling_auto import (
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
    from .modeling_bart import (
        BART_PRETRAINED_MODEL_ARCHIVE_LIST,
        BartForConditionalGeneration,
        BartForQuestionAnswering,
        BartForSequenceClassification,
        BartModel,
        PretrainedBartModel,
    )
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
    from .modeling_bert_generation import (
        BertGenerationDecoder,
        BertGenerationEncoder,
        load_tf_weights_in_bert_generation,
    )
    from .modeling_blenderbot import BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST, BlenderbotForConditionalGeneration
    from .modeling_camembert import (
        CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        CamembertForCausalLM,
        CamembertForMaskedLM,
        CamembertForMultipleChoice,
        CamembertForQuestionAnswering,
        CamembertForSequenceClassification,
        CamembertForTokenClassification,
        CamembertModel,
    )
    from .modeling_ctrl import CTRL_PRETRAINED_MODEL_ARCHIVE_LIST, CTRLLMHeadModel, CTRLModel, CTRLPreTrainedModel
    from .modeling_deberta import (
        DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        DebertaForSequenceClassification,
        DebertaModel,
        DebertaPreTrainedModel,
    )
    from .modeling_distilbert import (
        DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        DistilBertForMaskedLM,
        DistilBertForMultipleChoice,
        DistilBertForQuestionAnswering,
        DistilBertForSequenceClassification,
        DistilBertForTokenClassification,
        DistilBertModel,
        DistilBertPreTrainedModel,
    )
    from .modeling_dpr import (
        DPRContextEncoder,
        DPRPretrainedContextEncoder,
        DPRPretrainedQuestionEncoder,
        DPRPretrainedReader,
        DPRQuestionEncoder,
        DPRReader,
    )
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
    from .modeling_encoder_decoder import EncoderDecoderModel
    from .modeling_flaubert import (
        FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        FlaubertForMultipleChoice,
        FlaubertForQuestionAnswering,
        FlaubertForQuestionAnsweringSimple,
        FlaubertForSequenceClassification,
        FlaubertForTokenClassification,
        FlaubertModel,
        FlaubertWithLMHeadModel,
    )
    from .modeling_fsmt import FSMTForConditionalGeneration, FSMTModel, PretrainedFSMTModel
    from .modeling_funnel import (
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
    from .modeling_gpt2 import (
        GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
        GPT2DoubleHeadsModel,
        GPT2ForSequenceClassification,
        GPT2LMHeadModel,
        GPT2Model,
        GPT2PreTrainedModel,
        load_tf_weights_in_gpt2,
    )
    from .modeling_layoutlm import (
        LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,
        LayoutLMForMaskedLM,
        LayoutLMForTokenClassification,
        LayoutLMModel,
    )
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
    from .modeling_lxmert import (
        LxmertEncoder,
        LxmertForPreTraining,
        LxmertForQuestionAnswering,
        LxmertModel,
        LxmertPreTrainedModel,
        LxmertVisualFeatureEncoder,
        LxmertXLayer,
    )
    from .modeling_marian import MarianMTModel
    from .modeling_mbart import MBartForConditionalGeneration
    from .modeling_mmbt import MMBTForClassification, MMBTModel, ModalEmbeddings
    from .modeling_mobilebert import (
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
    from .modeling_openai import (
        OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,
        OpenAIGPTDoubleHeadsModel,
        OpenAIGPTForSequenceClassification,
        OpenAIGPTLMHeadModel,
        OpenAIGPTModel,
        OpenAIGPTPreTrainedModel,
        load_tf_weights_in_openai_gpt,
    )
    from .modeling_pegasus import PegasusForConditionalGeneration
    from .modeling_prophetnet import (
        PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST,
        ProphetNetDecoder,
        ProphetNetEncoder,
        ProphetNetForCausalLM,
        ProphetNetForConditionalGeneration,
        ProphetNetModel,
        ProphetNetPreTrainedModel,
    )
    from .modeling_rag import RagModel, RagSequenceForGeneration, RagTokenForGeneration
    from .modeling_reformer import (
        REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
        ReformerAttention,
        ReformerForMaskedLM,
        ReformerForQuestionAnswering,
        ReformerForSequenceClassification,
        ReformerLayer,
        ReformerModel,
        ReformerModelWithLMHead,
    )
    from .modeling_retribert import RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST, RetriBertModel, RetriBertPreTrainedModel
    from .modeling_roberta import (
        ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        RobertaForCausalLM,
        RobertaForMaskedLM,
        RobertaForMultipleChoice,
        RobertaForQuestionAnswering,
        RobertaForSequenceClassification,
        RobertaForTokenClassification,
        RobertaModel,
    )
    from .modeling_squeezebert import (
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
    from .modeling_t5 import (
        T5_PRETRAINED_MODEL_ARCHIVE_LIST,
        T5ForConditionalGeneration,
        T5Model,
        T5PreTrainedModel,
        load_tf_weights_in_t5,
    )
    from .modeling_transfo_xl import (
        TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
        AdaptiveEmbedding,
        TransfoXLLMHeadModel,
        TransfoXLModel,
        TransfoXLPreTrainedModel,
        load_tf_weights_in_transfo_xl,
    )
    from .modeling_utils import Conv1D, PreTrainedModel, apply_chunking_to_forward, prune_layer
    from .modeling_xlm import (
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
    from .modeling_xlm_prophetnet import (
        XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST,
        XLMProphetNetDecoder,
        XLMProphetNetEncoder,
        XLMProphetNetForCausalLM,
        XLMProphetNetForConditionalGeneration,
        XLMProphetNetModel,
    )
    from .modeling_xlm_roberta import (
        XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        XLMRobertaForCausalLM,
        XLMRobertaForMaskedLM,
        XLMRobertaForMultipleChoice,
        XLMRobertaForQuestionAnswering,
        XLMRobertaForSequenceClassification,
        XLMRobertaForTokenClassification,
        XLMRobertaModel,
    )
    from .modeling_xlnet import (
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
    from .modeling_tf_albert import (
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
    from .modeling_tf_auto import (
        TF_MODEL_FOR_CAUSAL_LM_MAPPING,
        TF_MODEL_FOR_MASKED_LM_MAPPING,
        TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
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
    from .modeling_tf_bart import TFBartForConditionalGeneration, TFBartModel
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
    from .modeling_tf_blenderbot import TFBlenderbotForConditionalGeneration
    from .modeling_tf_camembert import (
        TF_CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFCamembertForMaskedLM,
        TFCamembertForMultipleChoice,
        TFCamembertForQuestionAnswering,
        TFCamembertForSequenceClassification,
        TFCamembertForTokenClassification,
        TFCamembertModel,
    )
    from .modeling_tf_ctrl import (
        TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFCTRLLMHeadModel,
        TFCTRLModel,
        TFCTRLPreTrainedModel,
    )
    from .modeling_tf_distilbert import (
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
    from .modeling_tf_flaubert import (
        TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFFlaubertForMultipleChoice,
        TFFlaubertForQuestionAnsweringSimple,
        TFFlaubertForSequenceClassification,
        TFFlaubertForTokenClassification,
        TFFlaubertModel,
        TFFlaubertWithLMHeadModel,
    )
    from .modeling_tf_funnel import (
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
    from .modeling_tf_gpt2 import (
        TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFGPT2DoubleHeadsModel,
        TFGPT2LMHeadModel,
        TFGPT2MainLayer,
        TFGPT2Model,
        TFGPT2PreTrainedModel,
    )
    from .modeling_tf_longformer import (
        TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFLongformerForMaskedLM,
        TFLongformerForQuestionAnswering,
        TFLongformerModel,
        TFLongformerSelfAttention,
    )
    from .modeling_tf_lxmert import (
        TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFLxmertForPreTraining,
        TFLxmertMainLayer,
        TFLxmertModel,
        TFLxmertPreTrainedModel,
        TFLxmertVisualFeatureEncoder,
    )
    from .modeling_tf_marian import TFMarianMTModel
    from .modeling_tf_mbart import TFMBartForConditionalGeneration
    from .modeling_tf_mobilebert import (
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
    from .modeling_tf_openai import (
        TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFOpenAIGPTDoubleHeadsModel,
        TFOpenAIGPTLMHeadModel,
        TFOpenAIGPTMainLayer,
        TFOpenAIGPTModel,
        TFOpenAIGPTPreTrainedModel,
    )
    from .modeling_tf_pegasus import TFPegasusForConditionalGeneration
    from .modeling_tf_roberta import (
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
    from .modeling_tf_t5 import (
        TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFT5ForConditionalGeneration,
        TFT5Model,
        TFT5PreTrainedModel,
    )
    from .modeling_tf_transfo_xl import (
        TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFAdaptiveEmbedding,
        TFTransfoXLLMHeadModel,
        TFTransfoXLMainLayer,
        TFTransfoXLModel,
        TFTransfoXLPreTrainedModel,
    )
    from .modeling_tf_utils import TFPreTrainedModel, TFSequenceSummary, TFSharedEmbeddings, shape_list
    from .modeling_tf_xlm import (
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
    from .modeling_tf_xlm_roberta import (
        TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFXLMRobertaForMaskedLM,
        TFXLMRobertaForMultipleChoice,
        TFXLMRobertaForQuestionAnswering,
        TFXLMRobertaForSequenceClassification,
        TFXLMRobertaForTokenClassification,
        TFXLMRobertaModel,
    )
    from .modeling_tf_xlnet import (
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
    from .modeling_flax_bert import FlaxBertModel
    from .modeling_flax_roberta import FlaxRobertaModel
else:
    # Import the same objects as dummies to get them in the namespace.
    # They will raise an import error if the user tries to instantiate / use them.
    from .utils.dummy_flax_objects import *


if not is_tf_available() and not is_torch_available():
    logger.warning(
        "Neither PyTorch nor TensorFlow >= 2.0 have been found."
        "Models won't be available and only tokenizers, configuration"
        "and file/data utilities can be used."
    )
