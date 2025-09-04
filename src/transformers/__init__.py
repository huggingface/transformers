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

__version__ = "4.56.1"

from pathlib import Path
from typing import TYPE_CHECKING

# Check the dependencies satisfy the minimal versions required.
from . import dependency_versions_check
from .utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_essentia_available,
    is_g2p_en_available,
    is_librosa_available,
    is_mistral_common_available,
    is_pretty_midi_available,
)

# Note: the following symbols are deliberately exported with `as`
# so that mypy, pylint or other static linters can recognize them,
# given that they are not exported using `__all__` in this file.
from .utils import is_bitsandbytes_available as is_bitsandbytes_available
from .utils import is_flax_available as is_flax_available
from .utils import is_keras_nlp_available as is_keras_nlp_available
from .utils import is_scipy_available as is_scipy_available
from .utils import is_sentencepiece_available as is_sentencepiece_available
from .utils import is_speech_available as is_speech_available
from .utils import is_tensorflow_text_available as is_tensorflow_text_available
from .utils import is_tf_available as is_tf_available
from .utils import is_timm_available as is_timm_available
from .utils import is_tokenizers_available as is_tokenizers_available
from .utils import is_torch_available as is_torch_available
from .utils import is_torchaudio_available as is_torchaudio_available
from .utils import is_torchvision_available as is_torchvision_available
from .utils import is_vision_available as is_vision_available
from .utils import logging as logging
from .utils.import_utils import define_import_structure


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Base objects, independent of any specific backend
_import_structure = {
    "audio_utils": [],
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
        "DataCollatorForMultipleChoice",
        "DataCollatorForPermutationLanguageModeling",
        "DataCollatorForSeq2Seq",
        "DataCollatorForSOP",
        "DataCollatorForTokenClassification",
        "DataCollatorForWholeWordMask",
        "DataCollatorWithFlattening",
        "DataCollatorWithPadding",
        "DefaultDataCollator",
        "default_data_collator",
    ],
    "data.metrics": [],
    "data.processors": [],
    "debug_utils": [],
    "dependency_versions_check": [],
    "dependency_versions_table": [],
    "dynamic_module_utils": [],
    "feature_extraction_sequence_utils": ["SequenceFeatureExtractor"],
    "feature_extraction_utils": ["BatchFeature", "FeatureExtractionMixin"],
    "file_utils": [],
    "generation": [
        "AsyncTextIteratorStreamer",
        "CompileConfig",
        "GenerationConfig",
        "TextIteratorStreamer",
        "TextStreamer",
        "WatermarkingConfig",
    ],
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
        "is_swanlab_available",
        "is_tensorboard_available",
        "is_trackio_available",
        "is_wandb_available",
    ],
    "loss": [],
    "modelcard": ["ModelCard"],
    # Losses
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
    "onnx": [],
    "pipelines": [
        "AudioClassificationPipeline",
        "AutomaticSpeechRecognitionPipeline",
        "CsvPipelineDataFormat",
        "DepthEstimationPipeline",
        "DocumentQuestionAnsweringPipeline",
        "FeatureExtractionPipeline",
        "FillMaskPipeline",
        "ImageClassificationPipeline",
        "ImageFeatureExtractionPipeline",
        "ImageSegmentationPipeline",
        "ImageTextToTextPipeline",
        "ImageToImagePipeline",
        "ImageToTextPipeline",
        "JsonPipelineDataFormat",
        "KeypointMatchingPipeline",
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
        "is_faiss_available",
        "is_flax_available",
        "is_keras_nlp_available",
        "is_matplotlib_available",
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
        "is_torch_hpu_available",
        "is_torch_mlu_available",
        "is_torch_musa_available",
        "is_torch_neuroncore_available",
        "is_torch_npu_available",
        "is_torchvision_available",
        "is_torch_xla_available",
        "is_torch_xpu_available",
        "is_vision_available",
        "logging",
    ],
    "utils.quantization_config": [
        "AqlmConfig",
        "AutoRoundConfig",
        "AwqConfig",
        "BitNetQuantConfig",
        "BitsAndBytesConfig",
        "CompressedTensorsConfig",
        "EetqConfig",
        "FbgemmFp8Config",
        "FineGrainedFP8Config",
        "GPTQConfig",
        "HiggsConfig",
        "HqqConfig",
        "Mxfp4Config",
        "QuantoConfig",
        "QuarkConfig",
        "FPQuantConfig",
        "SpQRConfig",
        "TorchAoConfig",
        "VptqConfig",
    ],
    "video_utils": [],
}

# tokenizers-backed objects
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_tokenizers_objects

    _import_structure["utils.dummy_tokenizers_objects"] = [
        name for name in dir(dummy_tokenizers_objects) if not name.startswith("_")
    ]
else:
    # Fast tokenizers structure
    _import_structure["tokenization_utils_fast"] = ["PreTrainedTokenizerFast"]


try:
    if not (is_sentencepiece_available() and is_tokenizers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_sentencepiece_and_tokenizers_objects

    _import_structure["utils.dummy_sentencepiece_and_tokenizers_objects"] = [
        name for name in dir(dummy_sentencepiece_and_tokenizers_objects) if not name.startswith("_")
    ]
else:
    _import_structure["convert_slow_tokenizer"] = [
        "SLOW_TO_FAST_CONVERTERS",
        "convert_slow_tokenizer",
    ]

try:
    if not (is_mistral_common_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_mistral_common_objects

    _import_structure["utils.dummy_mistral_common_objects"] = [
        name for name in dir(dummy_mistral_common_objects) if not name.startswith("_")
    ]
else:
    _import_structure["tokenization_mistral_common"] = ["MistralCommonTokenizer"]

# Vision-specific objects
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_vision_objects

    _import_structure["utils.dummy_vision_objects"] = [
        name for name in dir(dummy_vision_objects) if not name.startswith("_")
    ]
else:
    _import_structure["image_processing_base"] = ["ImageProcessingMixin"]
    _import_structure["image_processing_utils"] = ["BaseImageProcessor"]
    _import_structure["image_utils"] = ["ImageFeatureExtractionMixin"]

try:
    if not is_torchvision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_torchvision_objects

    _import_structure["utils.dummy_torchvision_objects"] = [
        name for name in dir(dummy_torchvision_objects) if not name.startswith("_")
    ]
else:
    _import_structure["image_processing_utils_fast"] = ["BaseImageProcessorFast"]
    _import_structure["video_processing_utils"] = ["BaseVideoProcessor"]

# PyTorch-backed objects
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_pt_objects

    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]
else:
    _import_structure["model_debugging_utils"] = [
        "model_addition_debugger_context",
    ]
    _import_structure["activations"] = []
    _import_structure["cache_utils"] = [
        "CacheLayerMixin",
        "DynamicLayer",
        "StaticLayer",
        "SlidingWindowLayer",
        "ChunkedSlidingLayer",
        "QuantoQuantizedLayer",
        "HQQQuantizedLayer",
        "Cache",
        "DynamicCache",
        "EncoderDecoderCache",
        "HQQQuantizedCache",
        "HybridCache",
        "HybridChunkedCache",
        "OffloadedCache",
        "OffloadedStaticCache",
        "QuantizedCache",
        "QuantoQuantizedCache",
        "SinkCache",
        "SlidingWindowCache",
        "StaticCache",
    ]
    _import_structure["data.datasets"] = [
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
    _import_structure["generation"].extend(
        [
            "AlternatingCodebooksLogitsProcessor",
            "BayesianDetectorConfig",
            "BayesianDetectorModel",
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
            "GenerationMixin",
            "HammingDiversityLogitsProcessor",
            "InfNanRemoveLogitsProcessor",
            "LogitNormalization",
            "LogitsProcessor",
            "LogitsProcessorList",
            "MaxLengthCriteria",
            "MaxTimeCriteria",
            "MinLengthLogitsProcessor",
            "MinNewTokensLengthLogitsProcessor",
            "MinPLogitsWarper",
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
            "SynthIDTextWatermarkDetector",
            "SynthIDTextWatermarkingConfig",
            "SynthIDTextWatermarkLogitsProcessor",
            "TemperatureLogitsWarper",
            "TopKLogitsWarper",
            "TopPLogitsWarper",
            "TypicalLogitsWarper",
            "UnbatchedClassifierFreeGuidanceLogitsProcessor",
            "WatermarkDetector",
            "WatermarkLogitsProcessor",
            "WhisperTimeStampLogitsProcessor",
        ]
    )

    # PyTorch domain libraries integration
    _import_structure["integrations.executorch"] = [
        "TorchExportableModuleWithStaticCache",
        "convert_and_export_with_cache",
    ]

    _import_structure["modeling_flash_attention_utils"] = []
    _import_structure["modeling_layers"] = ["GradientCheckpointingLayer"]
    _import_structure["modeling_outputs"] = []
    _import_structure["modeling_rope_utils"] = ["ROPE_INIT_FUNCTIONS", "dynamic_rope_update"]
    _import_structure["modeling_utils"] = ["PreTrainedModel", "AttentionInterface"]
    _import_structure["masking_utils"] = ["AttentionMaskInterface"]
    _import_structure["optimization"] = [
        "Adafactor",
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
    _import_structure["pytorch_utils"] = [
        "Conv1D",
        "apply_chunking_to_forward",
        "prune_layer",
        "infer_device",
    ]
    _import_structure["sagemaker"] = []
    _import_structure["time_series_utils"] = []
    _import_structure["trainer"] = ["Trainer"]
    _import_structure["trainer_pt_utils"] = ["torch_distributed_zero_first"]
    _import_structure["trainer_seq2seq"] = ["Seq2SeqTrainer"]

# TensorFlow-backed objects
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_tf_objects

    _import_structure["utils.dummy_tf_objects"] = [name for name in dir(dummy_tf_objects) if not name.startswith("_")]
else:
    _import_structure["activations_tf"] = []
    _import_structure["generation"].extend(
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
    _import_structure["keras_callbacks"] = ["KerasMetricCallback", "PushToHubCallback"]
    _import_structure["modeling_tf_outputs"] = []
    _import_structure["modeling_tf_utils"] = [
        "TFPreTrainedModel",
        "TFSequenceSummary",
        "TFSharedEmbeddings",
        "shape_list",
    ]
    _import_structure["optimization_tf"] = [
        "AdamWeightDecay",
        "GradientAccumulator",
        "WarmUp",
        "create_optimizer",
    ]
    _import_structure["tf_utils"] = []


# FLAX-backed objects
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_flax_objects

    _import_structure["utils.dummy_flax_objects"] = [
        name for name in dir(dummy_flax_objects) if not name.startswith("_")
    ]
else:
    _import_structure["generation"].extend(
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
    _import_structure["modeling_flax_outputs"] = []
    _import_structure["modeling_flax_utils"] = ["FlaxPreTrainedModel"]

# Direct imports for type-checking
if TYPE_CHECKING:
    # All modeling imports
    from .cache_utils import Cache as Cache
    from .cache_utils import ChunkedSlidingLayer as ChunkedSlidingLayer
    from .cache_utils import DynamicCache as DynamicCache
    from .cache_utils import DynamicLayer as DynamicLayer
    from .cache_utils import EncoderDecoderCache as EncoderDecoderCache
    from .cache_utils import HQQQuantizedCache as HQQQuantizedCache
    from .cache_utils import HQQQuantizedLayer as HQQQuantizedLayer
    from .cache_utils import HybridCache as HybridCache
    from .cache_utils import MambaCache as MambaCache
    from .cache_utils import OffloadedCache as OffloadedCache
    from .cache_utils import OffloadedStaticCache as OffloadedStaticCache
    from .cache_utils import QuantizedCache as QuantizedCache
    from .cache_utils import QuantoQuantizedCache as QuantoQuantizedCache
    from .cache_utils import QuantoQuantizedLayer as QuantoQuantizedLayer
    from .cache_utils import SinkCache as SinkCache
    from .cache_utils import SlidingWindowCache as SlidingWindowCache
    from .cache_utils import SlidingWindowLayer as SlidingWindowLayer
    from .cache_utils import StaticCache as StaticCache
    from .cache_utils import StaticLayer as StaticLayer
    from .configuration_utils import PretrainedConfig as PretrainedConfig
    from .convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS as SLOW_TO_FAST_CONVERTERS
    from .convert_slow_tokenizer import convert_slow_tokenizer as convert_slow_tokenizer

    # Data
    from .data import DataProcessor as DataProcessor
    from .data import InputExample as InputExample
    from .data import InputFeatures as InputFeatures
    from .data import SingleSentenceClassificationProcessor as SingleSentenceClassificationProcessor
    from .data import SquadExample as SquadExample
    from .data import SquadFeatures as SquadFeatures
    from .data import SquadV1Processor as SquadV1Processor
    from .data import SquadV2Processor as SquadV2Processor
    from .data import glue_compute_metrics as glue_compute_metrics
    from .data import glue_convert_examples_to_features as glue_convert_examples_to_features
    from .data import glue_output_modes as glue_output_modes
    from .data import glue_processors as glue_processors
    from .data import glue_tasks_num_labels as glue_tasks_num_labels
    from .data import squad_convert_examples_to_features as squad_convert_examples_to_features
    from .data import xnli_compute_metrics as xnli_compute_metrics
    from .data import xnli_output_modes as xnli_output_modes
    from .data import xnli_processors as xnli_processors
    from .data import xnli_tasks_num_labels as xnli_tasks_num_labels
    from .data.data_collator import DataCollator as DataCollator
    from .data.data_collator import DataCollatorForLanguageModeling as DataCollatorForLanguageModeling
    from .data.data_collator import DataCollatorForMultipleChoice as DataCollatorForMultipleChoice
    from .data.data_collator import (
        DataCollatorForPermutationLanguageModeling as DataCollatorForPermutationLanguageModeling,
    )
    from .data.data_collator import DataCollatorForSeq2Seq as DataCollatorForSeq2Seq
    from .data.data_collator import DataCollatorForSOP as DataCollatorForSOP
    from .data.data_collator import DataCollatorForTokenClassification as DataCollatorForTokenClassification
    from .data.data_collator import DataCollatorForWholeWordMask as DataCollatorForWholeWordMask
    from .data.data_collator import DataCollatorWithFlattening as DataCollatorWithFlattening
    from .data.data_collator import DataCollatorWithPadding as DataCollatorWithPadding
    from .data.data_collator import DefaultDataCollator as DefaultDataCollator
    from .data.data_collator import default_data_collator as default_data_collator
    from .data.datasets import GlueDataset as GlueDataset
    from .data.datasets import GlueDataTrainingArguments as GlueDataTrainingArguments
    from .data.datasets import LineByLineTextDataset as LineByLineTextDataset
    from .data.datasets import LineByLineWithRefDataset as LineByLineWithRefDataset
    from .data.datasets import LineByLineWithSOPTextDataset as LineByLineWithSOPTextDataset
    from .data.datasets import SquadDataset as SquadDataset
    from .data.datasets import SquadDataTrainingArguments as SquadDataTrainingArguments
    from .data.datasets import TextDataset as TextDataset
    from .data.datasets import TextDatasetForNextSentencePrediction as TextDatasetForNextSentencePrediction
    from .feature_extraction_sequence_utils import SequenceFeatureExtractor as SequenceFeatureExtractor

    # Feature Extractor
    from .feature_extraction_utils import BatchFeature as BatchFeature
    from .feature_extraction_utils import FeatureExtractionMixin as FeatureExtractionMixin

    # Generation
    from .generation import AlternatingCodebooksLogitsProcessor as AlternatingCodebooksLogitsProcessor
    from .generation import AsyncTextIteratorStreamer as AsyncTextIteratorStreamer
    from .generation import BayesianDetectorConfig as BayesianDetectorConfig
    from .generation import BayesianDetectorModel as BayesianDetectorModel
    from .generation import BeamScorer as BeamScorer
    from .generation import BeamSearchScorer as BeamSearchScorer
    from .generation import ClassifierFreeGuidanceLogitsProcessor as ClassifierFreeGuidanceLogitsProcessor
    from .generation import CompileConfig as CompileConfig
    from .generation import ConstrainedBeamSearchScorer as ConstrainedBeamSearchScorer
    from .generation import Constraint as Constraint
    from .generation import ConstraintListState as ConstraintListState
    from .generation import DisjunctiveConstraint as DisjunctiveConstraint
    from .generation import EncoderNoRepeatNGramLogitsProcessor as EncoderNoRepeatNGramLogitsProcessor
    from .generation import EncoderRepetitionPenaltyLogitsProcessor as EncoderRepetitionPenaltyLogitsProcessor
    from .generation import EosTokenCriteria as EosTokenCriteria
    from .generation import EpsilonLogitsWarper as EpsilonLogitsWarper
    from .generation import EtaLogitsWarper as EtaLogitsWarper
    from .generation import ExponentialDecayLengthPenalty as ExponentialDecayLengthPenalty
    from .generation import FlaxForcedBOSTokenLogitsProcessor as FlaxForcedBOSTokenLogitsProcessor
    from .generation import FlaxForcedEOSTokenLogitsProcessor as FlaxForcedEOSTokenLogitsProcessor
    from .generation import FlaxForceTokensLogitsProcessor as FlaxForceTokensLogitsProcessor
    from .generation import FlaxGenerationMixin as FlaxGenerationMixin
    from .generation import FlaxLogitsProcessor as FlaxLogitsProcessor
    from .generation import FlaxLogitsProcessorList as FlaxLogitsProcessorList
    from .generation import FlaxLogitsWarper as FlaxLogitsWarper
    from .generation import FlaxMinLengthLogitsProcessor as FlaxMinLengthLogitsProcessor
    from .generation import FlaxSuppressTokensAtBeginLogitsProcessor as FlaxSuppressTokensAtBeginLogitsProcessor
    from .generation import FlaxSuppressTokensLogitsProcessor as FlaxSuppressTokensLogitsProcessor
    from .generation import FlaxTemperatureLogitsWarper as FlaxTemperatureLogitsWarper
    from .generation import FlaxTopKLogitsWarper as FlaxTopKLogitsWarper
    from .generation import FlaxTopPLogitsWarper as FlaxTopPLogitsWarper
    from .generation import FlaxWhisperTimeStampLogitsProcessor as FlaxWhisperTimeStampLogitsProcessor
    from .generation import ForcedBOSTokenLogitsProcessor as ForcedBOSTokenLogitsProcessor
    from .generation import ForcedEOSTokenLogitsProcessor as ForcedEOSTokenLogitsProcessor
    from .generation import GenerationConfig as GenerationConfig
    from .generation import GenerationMixin as GenerationMixin
    from .generation import HammingDiversityLogitsProcessor as HammingDiversityLogitsProcessor
    from .generation import InfNanRemoveLogitsProcessor as InfNanRemoveLogitsProcessor
    from .generation import LogitNormalization as LogitNormalization
    from .generation import LogitsProcessor as LogitsProcessor
    from .generation import LogitsProcessorList as LogitsProcessorList
    from .generation import MaxLengthCriteria as MaxLengthCriteria
    from .generation import MaxTimeCriteria as MaxTimeCriteria
    from .generation import MinLengthLogitsProcessor as MinLengthLogitsProcessor
    from .generation import MinNewTokensLengthLogitsProcessor as MinNewTokensLengthLogitsProcessor
    from .generation import MinPLogitsWarper as MinPLogitsWarper
    from .generation import NoBadWordsLogitsProcessor as NoBadWordsLogitsProcessor
    from .generation import NoRepeatNGramLogitsProcessor as NoRepeatNGramLogitsProcessor
    from .generation import PhrasalConstraint as PhrasalConstraint
    from .generation import PrefixConstrainedLogitsProcessor as PrefixConstrainedLogitsProcessor
    from .generation import RepetitionPenaltyLogitsProcessor as RepetitionPenaltyLogitsProcessor
    from .generation import SequenceBiasLogitsProcessor as SequenceBiasLogitsProcessor
    from .generation import StoppingCriteria as StoppingCriteria
    from .generation import StoppingCriteriaList as StoppingCriteriaList
    from .generation import StopStringCriteria as StopStringCriteria
    from .generation import SuppressTokensAtBeginLogitsProcessor as SuppressTokensAtBeginLogitsProcessor
    from .generation import SuppressTokensLogitsProcessor as SuppressTokensLogitsProcessor
    from .generation import SynthIDTextWatermarkDetector as SynthIDTextWatermarkDetector
    from .generation import SynthIDTextWatermarkingConfig as SynthIDTextWatermarkingConfig
    from .generation import SynthIDTextWatermarkLogitsProcessor as SynthIDTextWatermarkLogitsProcessor
    from .generation import TemperatureLogitsWarper as TemperatureLogitsWarper
    from .generation import TextIteratorStreamer as TextIteratorStreamer
    from .generation import TextStreamer as TextStreamer
    from .generation import TFForcedBOSTokenLogitsProcessor as TFForcedBOSTokenLogitsProcessor
    from .generation import TFForcedEOSTokenLogitsProcessor as TFForcedEOSTokenLogitsProcessor
    from .generation import TFForceTokensLogitsProcessor as TFForceTokensLogitsProcessor
    from .generation import TFGenerationMixin as TFGenerationMixin
    from .generation import TFLogitsProcessor as TFLogitsProcessor
    from .generation import TFLogitsProcessorList as TFLogitsProcessorList
    from .generation import TFLogitsWarper as TFLogitsWarper
    from .generation import TFMinLengthLogitsProcessor as TFMinLengthLogitsProcessor
    from .generation import TFNoBadWordsLogitsProcessor as TFNoBadWordsLogitsProcessor
    from .generation import TFNoRepeatNGramLogitsProcessor as TFNoRepeatNGramLogitsProcessor
    from .generation import TFRepetitionPenaltyLogitsProcessor as TFRepetitionPenaltyLogitsProcessor
    from .generation import TFSuppressTokensAtBeginLogitsProcessor as TFSuppressTokensAtBeginLogitsProcessor
    from .generation import TFSuppressTokensLogitsProcessor as TFSuppressTokensLogitsProcessor
    from .generation import TFTemperatureLogitsWarper as TFTemperatureLogitsWarper
    from .generation import TFTopKLogitsWarper as TFTopKLogitsWarper
    from .generation import TFTopPLogitsWarper as TFTopPLogitsWarper
    from .generation import TopKLogitsWarper as TopKLogitsWarper
    from .generation import TopPLogitsWarper as TopPLogitsWarper
    from .generation import TypicalLogitsWarper as TypicalLogitsWarper
    from .generation import (
        UnbatchedClassifierFreeGuidanceLogitsProcessor as UnbatchedClassifierFreeGuidanceLogitsProcessor,
    )
    from .generation import WatermarkDetector as WatermarkDetector
    from .generation import WatermarkingConfig as WatermarkingConfig
    from .generation import WatermarkLogitsProcessor as WatermarkLogitsProcessor
    from .generation import WhisperTimeStampLogitsProcessor as WhisperTimeStampLogitsProcessor
    from .hf_argparser import HfArgumentParser as HfArgumentParser
    from .image_processing_base import ImageProcessingMixin as ImageProcessingMixin
    from .image_processing_utils import BaseImageProcessor as BaseImageProcessor
    from .image_processing_utils_fast import BaseImageProcessorFast as BaseImageProcessorFast
    from .image_utils import ImageFeatureExtractionMixin as ImageFeatureExtractionMixin

    # Integrations
    from .integrations import is_clearml_available as is_clearml_available
    from .integrations import is_comet_available as is_comet_available
    from .integrations import is_dvclive_available as is_dvclive_available
    from .integrations import is_neptune_available as is_neptune_available
    from .integrations import is_optuna_available as is_optuna_available
    from .integrations import is_ray_available as is_ray_available
    from .integrations import is_ray_tune_available as is_ray_tune_available
    from .integrations import is_sigopt_available as is_sigopt_available
    from .integrations import is_swanlab_available as is_swanlab_available
    from .integrations import is_tensorboard_available as is_tensorboard_available
    from .integrations import is_trackio_available as is_trackio_available
    from .integrations import is_wandb_available as is_wandb_available
    from .integrations.executorch import TorchExportableModuleWithStaticCache as TorchExportableModuleWithStaticCache
    from .integrations.executorch import convert_and_export_with_cache as convert_and_export_with_cache
    from .keras_callbacks import KerasMetricCallback as KerasMetricCallback
    from .keras_callbacks import PushToHubCallback as PushToHubCallback
    from .masking_utils import AttentionMaskInterface as AttentionMaskInterface
    from .model_debugging_utils import model_addition_debugger_context as model_addition_debugger_context

    # Model Cards
    from .modelcard import ModelCard as ModelCard
    from .modeling_flax_utils import FlaxPreTrainedModel as FlaxPreTrainedModel
    from .modeling_layers import GradientCheckpointingLayer as GradientCheckpointingLayer
    from .modeling_rope_utils import ROPE_INIT_FUNCTIONS as ROPE_INIT_FUNCTIONS
    from .modeling_rope_utils import dynamic_rope_update as dynamic_rope_update

    # TF 2.0 <=> PyTorch conversion utilities
    from .modeling_tf_pytorch_utils import (
        convert_tf_weight_name_to_pt_weight_name as convert_tf_weight_name_to_pt_weight_name,
    )
    from .modeling_tf_pytorch_utils import load_pytorch_checkpoint_in_tf2_model as load_pytorch_checkpoint_in_tf2_model
    from .modeling_tf_pytorch_utils import load_pytorch_model_in_tf2_model as load_pytorch_model_in_tf2_model
    from .modeling_tf_pytorch_utils import load_pytorch_weights_in_tf2_model as load_pytorch_weights_in_tf2_model
    from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model as load_tf2_checkpoint_in_pytorch_model
    from .modeling_tf_pytorch_utils import load_tf2_model_in_pytorch_model as load_tf2_model_in_pytorch_model
    from .modeling_tf_pytorch_utils import load_tf2_weights_in_pytorch_model as load_tf2_weights_in_pytorch_model
    from .modeling_tf_utils import TFPreTrainedModel as TFPreTrainedModel
    from .modeling_tf_utils import TFSequenceSummary as TFSequenceSummary
    from .modeling_tf_utils import TFSharedEmbeddings as TFSharedEmbeddings
    from .modeling_tf_utils import shape_list as shape_list
    from .modeling_utils import AttentionInterface as AttentionInterface
    from .modeling_utils import PreTrainedModel as PreTrainedModel
    from .models import *
    from .models.timm_wrapper import TimmWrapperImageProcessor as TimmWrapperImageProcessor

    # Optimization
    from .optimization import Adafactor as Adafactor
    from .optimization import get_constant_schedule as get_constant_schedule
    from .optimization import get_constant_schedule_with_warmup as get_constant_schedule_with_warmup
    from .optimization import get_cosine_schedule_with_warmup as get_cosine_schedule_with_warmup
    from .optimization import (
        get_cosine_with_hard_restarts_schedule_with_warmup as get_cosine_with_hard_restarts_schedule_with_warmup,
    )
    from .optimization import get_inverse_sqrt_schedule as get_inverse_sqrt_schedule
    from .optimization import get_linear_schedule_with_warmup as get_linear_schedule_with_warmup
    from .optimization import get_polynomial_decay_schedule_with_warmup as get_polynomial_decay_schedule_with_warmup
    from .optimization import get_scheduler as get_scheduler
    from .optimization import get_wsd_schedule as get_wsd_schedule

    # Optimization
    from .optimization_tf import AdamWeightDecay as AdamWeightDecay
    from .optimization_tf import GradientAccumulator as GradientAccumulator
    from .optimization_tf import WarmUp as WarmUp
    from .optimization_tf import create_optimizer as create_optimizer

    # Pipelines
    from .pipelines import AudioClassificationPipeline as AudioClassificationPipeline
    from .pipelines import AutomaticSpeechRecognitionPipeline as AutomaticSpeechRecognitionPipeline
    from .pipelines import CsvPipelineDataFormat as CsvPipelineDataFormat
    from .pipelines import DepthEstimationPipeline as DepthEstimationPipeline
    from .pipelines import DocumentQuestionAnsweringPipeline as DocumentQuestionAnsweringPipeline
    from .pipelines import FeatureExtractionPipeline as FeatureExtractionPipeline
    from .pipelines import FillMaskPipeline as FillMaskPipeline
    from .pipelines import ImageClassificationPipeline as ImageClassificationPipeline
    from .pipelines import ImageFeatureExtractionPipeline as ImageFeatureExtractionPipeline
    from .pipelines import ImageSegmentationPipeline as ImageSegmentationPipeline
    from .pipelines import ImageTextToTextPipeline as ImageTextToTextPipeline
    from .pipelines import ImageToImagePipeline as ImageToImagePipeline
    from .pipelines import ImageToTextPipeline as ImageToTextPipeline
    from .pipelines import JsonPipelineDataFormat as JsonPipelineDataFormat
    from .pipelines import KeypointMatchingPipeline as KeypointMatchingPipeline
    from .pipelines import MaskGenerationPipeline as MaskGenerationPipeline
    from .pipelines import NerPipeline as NerPipeline
    from .pipelines import ObjectDetectionPipeline as ObjectDetectionPipeline
    from .pipelines import PipedPipelineDataFormat as PipedPipelineDataFormat
    from .pipelines import Pipeline as Pipeline
    from .pipelines import PipelineDataFormat as PipelineDataFormat
    from .pipelines import QuestionAnsweringPipeline as QuestionAnsweringPipeline
    from .pipelines import SummarizationPipeline as SummarizationPipeline
    from .pipelines import TableQuestionAnsweringPipeline as TableQuestionAnsweringPipeline
    from .pipelines import Text2TextGenerationPipeline as Text2TextGenerationPipeline
    from .pipelines import TextClassificationPipeline as TextClassificationPipeline
    from .pipelines import TextGenerationPipeline as TextGenerationPipeline
    from .pipelines import TextToAudioPipeline as TextToAudioPipeline
    from .pipelines import TokenClassificationPipeline as TokenClassificationPipeline
    from .pipelines import TranslationPipeline as TranslationPipeline
    from .pipelines import VideoClassificationPipeline as VideoClassificationPipeline
    from .pipelines import VisualQuestionAnsweringPipeline as VisualQuestionAnsweringPipeline
    from .pipelines import ZeroShotAudioClassificationPipeline as ZeroShotAudioClassificationPipeline
    from .pipelines import ZeroShotClassificationPipeline as ZeroShotClassificationPipeline
    from .pipelines import ZeroShotImageClassificationPipeline as ZeroShotImageClassificationPipeline
    from .pipelines import ZeroShotObjectDetectionPipeline as ZeroShotObjectDetectionPipeline
    from .pipelines import pipeline as pipeline
    from .processing_utils import ProcessorMixin as ProcessorMixin
    from .pytorch_utils import Conv1D as Conv1D
    from .pytorch_utils import apply_chunking_to_forward as apply_chunking_to_forward
    from .pytorch_utils import prune_layer as prune_layer

    # Tokenization
    from .tokenization_utils import PreTrainedTokenizer as PreTrainedTokenizer
    from .tokenization_utils_base import AddedToken as AddedToken
    from .tokenization_utils_base import BatchEncoding as BatchEncoding
    from .tokenization_utils_base import CharSpan as CharSpan
    from .tokenization_utils_base import PreTrainedTokenizerBase as PreTrainedTokenizerBase
    from .tokenization_utils_base import SpecialTokensMixin as SpecialTokensMixin
    from .tokenization_utils_base import TokenSpan as TokenSpan
    from .tokenization_utils_fast import PreTrainedTokenizerFast as PreTrainedTokenizerFast

    # Trainer
    from .trainer import Trainer as Trainer

    # Trainer
    from .trainer_callback import DefaultFlowCallback as DefaultFlowCallback
    from .trainer_callback import EarlyStoppingCallback as EarlyStoppingCallback
    from .trainer_callback import PrinterCallback as PrinterCallback
    from .trainer_callback import ProgressCallback as ProgressCallback
    from .trainer_callback import TrainerCallback as TrainerCallback
    from .trainer_callback import TrainerControl as TrainerControl
    from .trainer_callback import TrainerState as TrainerState
    from .trainer_pt_utils import torch_distributed_zero_first as torch_distributed_zero_first
    from .trainer_seq2seq import Seq2SeqTrainer as Seq2SeqTrainer
    from .trainer_utils import EvalPrediction as EvalPrediction
    from .trainer_utils import IntervalStrategy as IntervalStrategy
    from .trainer_utils import SchedulerType as SchedulerType
    from .trainer_utils import enable_full_determinism as enable_full_determinism
    from .trainer_utils import set_seed as set_seed
    from .training_args import TrainingArguments as TrainingArguments
    from .training_args_seq2seq import Seq2SeqTrainingArguments as Seq2SeqTrainingArguments
    from .training_args_tf import TFTrainingArguments as TFTrainingArguments

    # Files and general utilities
    from .utils import CONFIG_NAME as CONFIG_NAME
    from .utils import MODEL_CARD_NAME as MODEL_CARD_NAME
    from .utils import PYTORCH_PRETRAINED_BERT_CACHE as PYTORCH_PRETRAINED_BERT_CACHE
    from .utils import PYTORCH_TRANSFORMERS_CACHE as PYTORCH_TRANSFORMERS_CACHE
    from .utils import SPIECE_UNDERLINE as SPIECE_UNDERLINE
    from .utils import TF2_WEIGHTS_NAME as TF2_WEIGHTS_NAME
    from .utils import TF_WEIGHTS_NAME as TF_WEIGHTS_NAME
    from .utils import TRANSFORMERS_CACHE as TRANSFORMERS_CACHE
    from .utils import WEIGHTS_NAME as WEIGHTS_NAME
    from .utils import TensorType as TensorType
    from .utils import add_end_docstrings as add_end_docstrings
    from .utils import add_start_docstrings as add_start_docstrings
    from .utils import is_apex_available as is_apex_available
    from .utils import is_av_available as is_av_available
    from .utils import is_datasets_available as is_datasets_available
    from .utils import is_faiss_available as is_faiss_available
    from .utils import is_matplotlib_available as is_matplotlib_available
    from .utils import is_phonemizer_available as is_phonemizer_available
    from .utils import is_psutil_available as is_psutil_available
    from .utils import is_py3nvml_available as is_py3nvml_available
    from .utils import is_pyctcdecode_available as is_pyctcdecode_available
    from .utils import is_sacremoses_available as is_sacremoses_available
    from .utils import is_safetensors_available as is_safetensors_available
    from .utils import is_sklearn_available as is_sklearn_available
    from .utils import is_torch_hpu_available as is_torch_hpu_available
    from .utils import is_torch_mlu_available as is_torch_mlu_available
    from .utils import is_torch_musa_available as is_torch_musa_available
    from .utils import is_torch_neuroncore_available as is_torch_neuroncore_available
    from .utils import is_torch_npu_available as is_torch_npu_available
    from .utils import is_torch_xla_available as is_torch_xla_available
    from .utils import is_torch_xpu_available as is_torch_xpu_available
    from .utils import logging as logging

    # bitsandbytes config
    from .utils.quantization_config import AqlmConfig as AqlmConfig
    from .utils.quantization_config import AutoRoundConfig as AutoRoundConfig
    from .utils.quantization_config import AwqConfig as AwqConfig
    from .utils.quantization_config import BitNetQuantConfig as BitNetQuantConfig
    from .utils.quantization_config import BitsAndBytesConfig as BitsAndBytesConfig
    from .utils.quantization_config import CompressedTensorsConfig as CompressedTensorsConfig
    from .utils.quantization_config import EetqConfig as EetqConfig
    from .utils.quantization_config import FbgemmFp8Config as FbgemmFp8Config
    from .utils.quantization_config import FineGrainedFP8Config as FineGrainedFP8Config
    from .utils.quantization_config import FPQuantConfig as FPQuantConfig
    from .utils.quantization_config import GPTQConfig as GPTQConfig
    from .utils.quantization_config import HiggsConfig as HiggsConfig
    from .utils.quantization_config import HqqConfig as HqqConfig
    from .utils.quantization_config import QuantoConfig as QuantoConfig
    from .utils.quantization_config import QuarkConfig as QuarkConfig
    from .utils.quantization_config import SpQRConfig as SpQRConfig
    from .utils.quantization_config import TorchAoConfig as TorchAoConfig
    from .utils.quantization_config import VptqConfig as VptqConfig
    from .video_processing_utils import BaseVideoProcessor as BaseVideoProcessor

else:
    import sys

    _import_structure = {k: set(v) for k, v in _import_structure.items()}

    import_structure = define_import_structure(Path(__file__).parent / "models", prefix="models")
    import_structure[frozenset({})].update(_import_structure)

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )


if not is_tf_available() and not is_torch_available() and not is_flax_available():
    logger.warning_advice(
        "None of PyTorch, TensorFlow >= 2.0, or Flax have been found. "
        "Models won't be available and only tokenizers, configuration "
        "and file/data utilities can be used."
    )
