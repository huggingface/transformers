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

# When adding a new object to this init, remember to add it twice: once inside the `_import_structure` dictionary and
# once inside the `if TYPE_CHECKING` branch. The `TYPE_CHECKING` should have import statements as usual, but they are
# only there for type checking. The `_import_structure` is a dictionary submodule to list of object names, and is used
# to defer the actual importing for when the objects are requested. This way `import transformers` provides the names
# in the namespace without actually importing anything (and especially none of the backends).

__version__ = "4.42.0.dev0"

import json
import os.path
from typing import TYPE_CHECKING

# Check the dependencies satisfy the minimal versions required.
from . import dependency_versions_check
from .utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_bitsandbytes_available,
    is_essentia_available,
    is_flax_available,
    is_g2p_en_available,
    is_keras_nlp_available,
    is_librosa_available,
    is_pretty_midi_available,
    is_scipy_available,
    is_sentencepiece_available,
    is_speech_available,
    is_tensorflow_text_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_torchaudio_available,
    is_torchvision_available,
    is_vision_available,
    logging,
)
from .utils.import_utils import define_import_structure, spread_import_structure

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Base objects, independent of any specific backend
_import_structure = {
    (): {
        "agents": [
            "Agent",
            "CodeAgent",
            "HfEngine",
            "PipelineTool",
            "ReactAgent",
            "ReactCodeAgent",
            "ReactJsonAgent",
            "Tool",
            "Toolbox",
            "ToolCollection",
            "launch_gradio_demo",
            "load_tool",
        ],
        "audio_utils": [],
        "benchmark": [],
        "commands": [],
        "configuration_utils": ["PretrainedConfig"],
        "convert_graph_to_onnx": [],
        "convert_slow_tokenizers_checkpoints_to_fast": [],
        "convert_tf_hub_seq_to_seq_bert_to_pytorch": [],
        "data": [
            "DataProcessor",
            "InputExample",
            "InputFeatures",
            "SingleSentenceClassificationProcessor",
            "SquadExample",
            "SquadFeatures",
            "SquadV1Processor",
            "SquadV2Processor",
            "glue_compute_metrics",
            "glue_convert_examples_to_features",
            "glue_output_modes",
            "glue_processors",
            "glue_tasks_num_labels",
            "squad_convert_examples_to_features",
            "xnli_compute_metrics",
            "xnli_output_modes",
            "xnli_processors",
            "xnli_tasks_num_labels",
        ],
        "data.data_collator": [
            "DataCollator",
            "DataCollatorForLanguageModeling",
            "DataCollatorForPermutationLanguageModeling",
            "DataCollatorForSeq2Seq",
            "DataCollatorForSOP",
            "DataCollatorForTokenClassification",
            "DataCollatorForWholeWordMask",
            "DataCollatorWithPadding",
            "DefaultDataCollator",
            "default_data_collator",
        ],
        "data.metrics": [],
        "data.processors": [],
        "debug_utils": [],
        "deepspeed": [],
        "dependency_versions_check": [],
        "dependency_versions_table": [],
        "dynamic_module_utils": [],
        "feature_extraction_sequence_utils": ["SequenceFeatureExtractor"],
        "feature_extraction_utils": ["BatchFeature", "FeatureExtractionMixin"],
        "file_utils": [],
        "generation": ["GenerationConfig", "TextIteratorStreamer", "TextStreamer"],
        "hf_argparser": ["HfArgumentParser"],
        "hyperparameter_search": [],
        "image_transforms": [],
        "integrations": [
            "is_clearml_available",
            "is_comet_available",
            "is_dvclive_available",
            "is_neptune_available",
            "is_optuna_available",
            "is_ray_available",
            "is_ray_tune_available",
            "is_sigopt_available",
            "is_tensorboard_available",
            "is_wandb_available",
        ],
        "modelcard": ["ModelCard"],
        "modeling_tf_pytorch_utils": [
            "convert_tf_weight_name_to_pt_weight_name",
            "load_pytorch_checkpoint_in_tf2_model",
            "load_pytorch_model_in_tf2_model",
            "load_pytorch_weights_in_tf2_model",
            "load_tf2_checkpoint_in_pytorch_model",
            "load_tf2_model_in_pytorch_model",
            "load_tf2_weights_in_pytorch_model",
        ],
        # Models
        "models": [],
        "onnx": [],
        "pipelines": [
            "AudioClassificationPipeline",
            "AutomaticSpeechRecognitionPipeline",
            "Conversation",
            "ConversationalPipeline",
            "CsvPipelineDataFormat",
            "DepthEstimationPipeline",
            "DocumentQuestionAnsweringPipeline",
            "FeatureExtractionPipeline",
            "FillMaskPipeline",
            "ImageClassificationPipeline",
            "ImageFeatureExtractionPipeline",
            "ImageSegmentationPipeline",
            "ImageToImagePipeline",
            "ImageToTextPipeline",
            "JsonPipelineDataFormat",
            "MaskGenerationPipeline",
            "NerPipeline",
            "ObjectDetectionPipeline",
            "PipedPipelineDataFormat",
            "Pipeline",
            "PipelineDataFormat",
            "QuestionAnsweringPipeline",
            "SummarizationPipeline",
            "TableQuestionAnsweringPipeline",
            "Text2TextGenerationPipeline",
            "TextClassificationPipeline",
            "TextGenerationPipeline",
            "TextToAudioPipeline",
            "TokenClassificationPipeline",
            "TranslationPipeline",
            "VideoClassificationPipeline",
            "VisualQuestionAnsweringPipeline",
            "ZeroShotAudioClassificationPipeline",
            "ZeroShotClassificationPipeline",
            "ZeroShotImageClassificationPipeline",
            "ZeroShotObjectDetectionPipeline",
            "pipeline",
        ],
        "processing_utils": ["ProcessorMixin"],
        "quantizers": [],
        "testing_utils": [],
        "tokenization_utils": ["PreTrainedTokenizer"],
        "tokenization_utils_base": [
            "AddedToken",
            "BatchEncoding",
            "CharSpan",
            "PreTrainedTokenizerBase",
            "SpecialTokensMixin",
            "TokenSpan",
        ],
        "trainer_callback": [
            "DefaultFlowCallback",
            "EarlyStoppingCallback",
            "PrinterCallback",
            "ProgressCallback",
            "TrainerCallback",
            "TrainerControl",
            "TrainerState",
        ],
        "trainer_utils": [
            "EvalPrediction",
            "IntervalStrategy",
            "SchedulerType",
            "enable_full_determinism",
            "set_seed",
        ],
        "training_args": ["TrainingArguments"],
        "training_args_seq2seq": ["Seq2SeqTrainingArguments"],
        "training_args_tf": ["TFTrainingArguments"],
        "utils": [
            "CONFIG_NAME",
            "MODEL_CARD_NAME",
            "PYTORCH_PRETRAINED_BERT_CACHE",
            "PYTORCH_TRANSFORMERS_CACHE",
            "SPIECE_UNDERLINE",
            "TF2_WEIGHTS_NAME",
            "TF_WEIGHTS_NAME",
            "TRANSFORMERS_CACHE",
            "WEIGHTS_NAME",
            "TensorType",
            "add_end_docstrings",
            "add_start_docstrings",
            "is_apex_available",
            "is_av_available",
            "is_bitsandbytes_available",
            "is_datasets_available",
            "is_decord_available",
            "is_faiss_available",
            "is_flax_available",
            "is_keras_nlp_available",
            "is_phonemizer_available",
            "is_psutil_available",
            "is_py3nvml_available",
            "is_pyctcdecode_available",
            "is_sacremoses_available",
            "is_safetensors_available",
            "is_scipy_available",
            "is_sentencepiece_available",
            "is_sklearn_available",
            "is_speech_available",
            "is_tensorflow_text_available",
            "is_tf_available",
            "is_timm_available",
            "is_tokenizers_available",
            "is_torch_available",
            "is_torch_mlu_available",
            "is_torch_neuroncore_available",
            "is_torch_npu_available",
            "is_torch_tpu_available",
            "is_torchvision_available",
            "is_torch_xla_available",
            "is_torch_xpu_available",
            "is_vision_available",
            "logging",
        ],
        "utils.quantization_config": [
            "AqlmConfig",
            "AwqConfig",
            "BitsAndBytesConfig",
            "EetqConfig",
            "GPTQConfig",
            "HqqConfig",
            "QuantoConfig",
        ],
    }
}

_import_structure.update(spread_import_structure(define_import_structure(os.path.dirname(__file__))))

_import_structure[('tokenizers',)]["tokenization_utils_fast"] = ["PreTrainedTokenizerFast"]
_import_structure[('sentencepiece', 'tokenizers')] = {
    "convert_slow_tokenizer": [
        "SLOW_TO_FAST_CONVERTERS",
        "convert_slow_tokenizer",
    ]
}
_import_structure[('vision',)]["image_processing_utils"] = ["ImageProcessingMixin"]
_import_structure[('vision',)]["image_utils"] = ["ImageFeatureExtractionMixin"]


_import_structure[('torch',)]["activations"] = []
_import_structure[('torch',)]["benchmark.benchmark"] = ["PyTorchBenchmark"]
_import_structure[('torch',)]["benchmark.benchmark_args"] = ["PyTorchBenchmarkArguments"]
_import_structure[('torch',)]["cache_utils"] = ["Cache", "DynamicCache", "SinkCache", "StaticCache"]
_import_structure[('torch',)]["data.datasets"] = [
    "GlueDataset",
    "GlueDataTrainingArguments",
    "LineByLineTextDataset",
    "LineByLineWithRefDataset",
    "LineByLineWithSOPTextDataset",
    "SquadDataset",
    "SquadDataTrainingArguments",
    "TextDataset",
    "TextDatasetForNextSentencePrediction",
]
_import_structure[('torch',)]["generation"] = (
    [
        "AlternatingCodebooksLogitsProcessor",
        "BeamScorer",
        "BeamSearchScorer",
        "ClassifierFreeGuidanceLogitsProcessor",
        "ConstrainedBeamSearchScorer",
        "Constraint",
        "ConstraintListState",
        "DisjunctiveConstraint",
        "EncoderNoRepeatNGramLogitsProcessor",
        "EncoderRepetitionPenaltyLogitsProcessor",
        "EosTokenCriteria",
        "EpsilonLogitsWarper",
        "EtaLogitsWarper",
        "ExponentialDecayLengthPenalty",
        "ForcedBOSTokenLogitsProcessor",
        "ForcedEOSTokenLogitsProcessor",
        "ForceTokensLogitsProcessor",
        "GenerationMixin",
        "HammingDiversityLogitsProcessor",
        "InfNanRemoveLogitsProcessor",
        "LogitNormalization",
        "LogitsProcessor",
        "LogitsProcessorList",
        "LogitsWarper",
        "MaxLengthCriteria",
        "MaxTimeCriteria",
        "MinLengthLogitsProcessor",
        "MinNewTokensLengthLogitsProcessor",
        "NoBadWordsLogitsProcessor",
        "NoRepeatNGramLogitsProcessor",
        "PhrasalConstraint",
        "PrefixConstrainedLogitsProcessor",
        "RepetitionPenaltyLogitsProcessor",
        "SequenceBiasLogitsProcessor",
        "StoppingCriteria",
        "StoppingCriteriaList",
        "StopStringCriteria",
        "SuppressTokensAtBeginLogitsProcessor",
        "SuppressTokensLogitsProcessor",
        "TemperatureLogitsWarper",
        "TopKLogitsWarper",
        "TopPLogitsWarper",
        "TypicalLogitsWarper",
        "UnbatchedClassifierFreeGuidanceLogitsProcessor",
        "WhisperTimeStampLogitsProcessor",
    ]
)
_import_structure[('torch',)]["modeling_outputs"] = []
_import_structure[('torch',)]["modeling_utils"] = ["PreTrainedModel"]

# PyTorch models structure



_import_structure[('torch',)]["optimization"] = [
    "Adafactor",
    "AdamW",
    "get_constant_schedule",
    "get_constant_schedule_with_warmup",
    "get_cosine_schedule_with_warmup",
    "get_cosine_with_hard_restarts_schedule_with_warmup",
    "get_inverse_sqrt_schedule",
    "get_linear_schedule_with_warmup",
    "get_polynomial_decay_schedule_with_warmup",
    "get_scheduler",
    "get_wsd_schedule",
]
_import_structure[('torch',)]["pytorch_utils"] = [
    "Conv1D",
    "apply_chunking_to_forward",
    "prune_layer",
]
_import_structure[('torch',)]["sagemaker"] = []
_import_structure[('torch',)]["time_series_utils"] = []
_import_structure[('torch',)]["trainer"] = ["Trainer"]
_import_structure[('torch',)]["trainer_pt_utils"] = ["torch_distributed_zero_first"]
_import_structure[('torch',)]["trainer_seq2seq"] = ["Seq2SeqTrainer"]

_import_structure[('tf',)]["activations_tf"] = []
_import_structure[('tf',)]["benchmark.benchmark_args_tf"] = ["TensorFlowBenchmarkArguments"]
_import_structure[('tf',)]["benchmark.benchmark_tf"] = ["TensorFlowBenchmark"]
_import_structure[('tf',)]["generation"] = (
    [
        "TFForcedBOSTokenLogitsProcessor",
        "TFForcedEOSTokenLogitsProcessor",
        "TFForceTokensLogitsProcessor",
        "TFGenerationMixin",
        "TFLogitsProcessor",
        "TFLogitsProcessorList",
        "TFLogitsWarper",
        "TFMinLengthLogitsProcessor",
        "TFNoBadWordsLogitsProcessor",
        "TFNoRepeatNGramLogitsProcessor",
        "TFRepetitionPenaltyLogitsProcessor",
        "TFSuppressTokensAtBeginLogitsProcessor",
        "TFSuppressTokensLogitsProcessor",
        "TFTemperatureLogitsWarper",
        "TFTopKLogitsWarper",
        "TFTopPLogitsWarper",
    ]
)
_import_structure[('tf',)]["keras_callbacks"] = ["KerasMetricCallback", "PushToHubCallback"]
_import_structure[('tf',)]["modeling_tf_outputs"] = []
_import_structure[('tf',)]["modeling_tf_utils"] = [
    "TFPreTrainedModel",
    "TFSequenceSummary",
    "TFSharedEmbeddings",
    "shape_list",
]
# TensorFlow models structure
_import_structure[('tf',)]["optimization_tf"] = [
    "AdamWeightDecay",
    "GradientAccumulator",
    "WarmUp",
    "create_optimizer",
]
_import_structure[('tf',)]["tf_utils"] = []


# FLAX-backed objects
_import_structure[('flax',)]["generation"] = (
    [
        "FlaxForcedBOSTokenLogitsProcessor",
        "FlaxForcedEOSTokenLogitsProcessor",
        "FlaxForceTokensLogitsProcessor",
        "FlaxGenerationMixin",
        "FlaxLogitsProcessor",
        "FlaxLogitsProcessorList",
        "FlaxLogitsWarper",
        "FlaxMinLengthLogitsProcessor",
        "FlaxTemperatureLogitsWarper",
        "FlaxSuppressTokensAtBeginLogitsProcessor",
        "FlaxSuppressTokensLogitsProcessor",
        "FlaxTopKLogitsWarper",
        "FlaxTopPLogitsWarper",
        "FlaxWhisperTimeStampLogitsProcessor",
    ]
)
_import_structure[('flax',)]["modeling_flax_outputs"] = []
_import_structure[('flax',)]["modeling_flax_utils"] = ["FlaxPreTrainedModel"]

# Direct imports for type-checking
if TYPE_CHECKING:
    # Configuration
    # Agents
    from .agents import (
        Agent,
        CodeAgent,
        HfEngine,
        PipelineTool,
        ReactAgent,
        ReactCodeAgent,
        ReactJsonAgent,
        Tool,
        Toolbox,
        ToolCollection,
        launch_gradio_demo,
        load_tool,
    )
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
    from .data.data_collator import (
        DataCollator,
        DataCollatorForLanguageModeling,
        DataCollatorForPermutationLanguageModeling,
        DataCollatorForSeq2Seq,
        DataCollatorForSOP,
        DataCollatorForTokenClassification,
        DataCollatorForWholeWordMask,
        DataCollatorWithPadding,
        DefaultDataCollator,
        default_data_collator,
    )
    from .feature_extraction_sequence_utils import SequenceFeatureExtractor

    # Feature Extractor
    from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin

    # Generation
    from .generation import GenerationConfig, TextIteratorStreamer, TextStreamer, WatermarkingConfig
    from .hf_argparser import HfArgumentParser

    # Integrations
    from .integrations import (
        is_clearml_available,
        is_comet_available,
        is_dvclive_available,
        is_neptune_available,
        is_optuna_available,
        is_ray_available,
        is_ray_tune_available,
        is_sigopt_available,
        is_tensorboard_available,
        is_wandb_available,
    )

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
        AudioClassificationPipeline,
        AutomaticSpeechRecognitionPipeline,
        Conversation,
        ConversationalPipeline,
        CsvPipelineDataFormat,
        DepthEstimationPipeline,
        DocumentQuestionAnsweringPipeline,
        FeatureExtractionPipeline,
        FillMaskPipeline,
        ImageClassificationPipeline,
        ImageFeatureExtractionPipeline,
        ImageSegmentationPipeline,
        ImageToImagePipeline,
        ImageToTextPipeline,
        JsonPipelineDataFormat,
        MaskGenerationPipeline,
        NerPipeline,
        ObjectDetectionPipeline,
        PipedPipelineDataFormat,
        Pipeline,
        PipelineDataFormat,
        QuestionAnsweringPipeline,
        SummarizationPipeline,
        TableQuestionAnsweringPipeline,
        Text2TextGenerationPipeline,
        TextClassificationPipeline,
        TextGenerationPipeline,
        TextToAudioPipeline,
        TokenClassificationPipeline,
        TranslationPipeline,
        VideoClassificationPipeline,
        VisualQuestionAnsweringPipeline,
        ZeroShotAudioClassificationPipeline,
        ZeroShotClassificationPipeline,
        ZeroShotImageClassificationPipeline,
        ZeroShotObjectDetectionPipeline,
        pipeline,
    )
    from .processing_utils import ProcessorMixin

    # Tokenization
    from .tokenization_utils import PreTrainedTokenizer
    from .tokenization_utils_base import (
        AddedToken,
        BatchEncoding,
        CharSpan,
        PreTrainedTokenizerBase,
        SpecialTokensMixin,
        TokenSpan,
    )

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
    from .trainer_utils import (
        EvalPrediction,
        IntervalStrategy,
        SchedulerType,
        enable_full_determinism,
        set_seed,
    )
    from .training_args import TrainingArguments
    from .training_args_seq2seq import Seq2SeqTrainingArguments
    from .training_args_tf import TFTrainingArguments

    # Files and general utilities
    from .utils import (
        CONFIG_NAME,
        MODEL_CARD_NAME,
        PYTORCH_PRETRAINED_BERT_CACHE,
        PYTORCH_TRANSFORMERS_CACHE,
        SPIECE_UNDERLINE,
        TF2_WEIGHTS_NAME,
        TF_WEIGHTS_NAME,
        TRANSFORMERS_CACHE,
        WEIGHTS_NAME,
        TensorType,
        add_end_docstrings,
        add_start_docstrings,
        is_apex_available,
        is_av_available,
        is_bitsandbytes_available,
        is_datasets_available,
        is_decord_available,
        is_faiss_available,
        is_flax_available,
        is_keras_nlp_available,
        is_phonemizer_available,
        is_psutil_available,
        is_py3nvml_available,
        is_pyctcdecode_available,
        is_sacremoses_available,
        is_safetensors_available,
        is_scipy_available,
        is_sentencepiece_available,
        is_sklearn_available,
        is_speech_available,
        is_tensorflow_text_available,
        is_tf_available,
        is_timm_available,
        is_tokenizers_available,
        is_torch_available,
        is_torch_mlu_available,
        is_torch_neuroncore_available,
        is_torch_npu_available,
        is_torch_tpu_available,
        is_torch_xla_available,
        is_torch_xpu_available,
        is_torchvision_available,
        is_vision_available,
        logging,
    )

    # bitsandbytes config
    from .utils.quantization_config import (
        AqlmConfig,
        AwqConfig,
        BitsAndBytesConfig,
        EetqConfig,
        GPTQConfig,
        HqqConfig,
        QuantoConfig,
    )

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_tokenizers_objects import *
    else:
        # Fast tokenizers imports
        from .tokenization_utils_fast import PreTrainedTokenizerFast

    try:
        if not (is_sentencepiece_available() and is_tokenizers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummies_sentencepiece_and_tokenizers_objects import *
    else:
        from .convert_slow_tokenizer import (
            SLOW_TO_FAST_CONVERTERS,
            convert_slow_tokenizer,
        )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_vision_objects import *
    else:
        from .image_processing_utils import ImageProcessingMixin
        from .image_utils import ImageFeatureExtractionMixin

    # Modeling
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torch_objects import *
    else:
        # Benchmarks
        from .benchmark.benchmark import PyTorchBenchmark
        from .benchmark.benchmark_args import PyTorchBenchmarkArguments
        from .cache_utils import Cache, DynamicCache, SinkCache, StaticCache
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
        from .generation import (
            AlternatingCodebooksLogitsProcessor,
            BeamScorer,
            BeamSearchScorer,
            ClassifierFreeGuidanceLogitsProcessor,
            ConstrainedBeamSearchScorer,
            Constraint,
            ConstraintListState,
            DisjunctiveConstraint,
            EncoderNoRepeatNGramLogitsProcessor,
            EncoderRepetitionPenaltyLogitsProcessor,
            EosTokenCriteria,
            EpsilonLogitsWarper,
            EtaLogitsWarper,
            ExponentialDecayLengthPenalty,
            ForcedBOSTokenLogitsProcessor,
            ForcedEOSTokenLogitsProcessor,
            ForceTokensLogitsProcessor,
            GenerationMixin,
            HammingDiversityLogitsProcessor,
            InfNanRemoveLogitsProcessor,
            LogitNormalization,
            LogitsProcessor,
            LogitsProcessorList,
            LogitsWarper,
            MaxLengthCriteria,
            MaxTimeCriteria,
            MinLengthLogitsProcessor,
            MinNewTokensLengthLogitsProcessor,
            MinPLogitsWarper,
            NoBadWordsLogitsProcessor,
            NoRepeatNGramLogitsProcessor,
            PhrasalConstraint,
            PrefixConstrainedLogitsProcessor,
            RepetitionPenaltyLogitsProcessor,
            SequenceBiasLogitsProcessor,
            StoppingCriteria,
            StoppingCriteriaList,
            StopStringCriteria,
            SuppressTokensAtBeginLogitsProcessor,
            SuppressTokensLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
            TypicalLogitsWarper,
            UnbatchedClassifierFreeGuidanceLogitsProcessor,
            WatermarkDetector,
            WatermarkLogitsProcessor,
            WhisperTimeStampLogitsProcessor,
        )
        from .modeling_utils import PreTrainedModel

        # PyTorch model imports

        # Optimization
        from .optimization import (
            Adafactor,
            AdamW,
            get_constant_schedule,
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
            get_cosine_with_hard_restarts_schedule_with_warmup,
            get_inverse_sqrt_schedule,
            get_linear_schedule_with_warmup,
            get_polynomial_decay_schedule_with_warmup,
            get_scheduler,
            get_wsd_schedule,
        )
        from .pytorch_utils import Conv1D, apply_chunking_to_forward, prune_layer

        # Trainer
        from .trainer import Trainer
        from .trainer_pt_utils import torch_distributed_zero_first
        from .trainer_seq2seq import Seq2SeqTrainer

    # TensorFlow
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # Import the same objects as dummies to get them in the namespace.
        # They will raise an import error if the user tries to instantiate / use them.
        from .utils.dummy_tf_objects import *
    else:
        from .benchmark.benchmark_args_tf import TensorFlowBenchmarkArguments

        # Benchmarks
        from .benchmark.benchmark_tf import TensorFlowBenchmark
        from .generation import (
            TFForcedBOSTokenLogitsProcessor,
            TFForcedEOSTokenLogitsProcessor,
            TFForceTokensLogitsProcessor,
            TFGenerationMixin,
            TFLogitsProcessor,
            TFLogitsProcessorList,
            TFLogitsWarper,
            TFMinLengthLogitsProcessor,
            TFNoBadWordsLogitsProcessor,
            TFNoRepeatNGramLogitsProcessor,
            TFRepetitionPenaltyLogitsProcessor,
            TFSuppressTokensAtBeginLogitsProcessor,
            TFSuppressTokensLogitsProcessor,
            TFTemperatureLogitsWarper,
            TFTopKLogitsWarper,
            TFTopPLogitsWarper,
        )
        from .keras_callbacks import KerasMetricCallback, PushToHubCallback
        from .modeling_tf_utils import (
            TFPreTrainedModel,
            TFSequenceSummary,
            TFSharedEmbeddings,
            shape_list,
        )

        # TensorFlow model imports

        # Optimization
        from .optimization_tf import (
            AdamWeightDecay,
            GradientAccumulator,
            WarmUp,
            create_optimizer,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # Import the same objects as dummies to get them in the namespace.
        # They will raise an import error if the user tries to instantiate / use them.
        from .utils.dummy_flax_objects import *
    else:
        from .generation import (
            FlaxForcedBOSTokenLogitsProcessor,
            FlaxForcedEOSTokenLogitsProcessor,
            FlaxForceTokensLogitsProcessor,
            FlaxGenerationMixin,
            FlaxLogitsProcessor,
            FlaxLogitsProcessorList,
            FlaxLogitsWarper,
            FlaxMinLengthLogitsProcessor,
            FlaxSuppressTokensAtBeginLogitsProcessor,
            FlaxSuppressTokensLogitsProcessor,
            FlaxTemperatureLogitsWarper,
            FlaxTopKLogitsWarper,
            FlaxTopPLogitsWarper,
            FlaxWhisperTimeStampLogitsProcessor,
        )
        from .modeling_flax_utils import FlaxPreTrainedModel

    from .models import *

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )


if not is_tf_available() and not is_torch_available() and not is_flax_available():
    logger.warning_advice(
        "None of PyTorch, TensorFlow >= 2.0, or Flax have been found. "
        "Models won't be available and only tokenizers, configuration "
        "and file/data utilities can be used."
    )
