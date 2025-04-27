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

__version__ = "4.52.0.dev0"

from pathlib import Path
from typing import TYPE_CHECKING

# Check the dependencies satisfy the minimal versions required.
from .utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_torchvision_available,
    is_vision_available,
    logging,
)
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
    "models.hindi_causal_lm": [
        "HindiCausalLMConfig",
        "HindiCausalLMTokenizer",
        "HindiCausalLMModel",
        "HindiCausalLMForCausalLM",
        "HindiCausalLMPreTrainedModel",
    ],
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
        "BitNetConfig",
        "BitsAndBytesConfig",
        "CompressedTensorsConfig",
        "EetqConfig",
        "FbgemmFp8Config",
        "FineGrainedFP8Config",
        "GPTQConfig",
        "HiggsConfig",
        "HqqConfig",
        "QuantoConfig",
        "QuarkConfig",
        "SpQRConfig",
        "TorchAoConfig",
        "VptqConfig",
    ],
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
        "Cache",
        "CacheConfig",
        "DynamicCache",
        "EncoderDecoderCache",
        "HQQQuantizedCache",
        "HybridCache",
        "MambaCache",
        "OffloadedCache",
        "OffloadedStaticCache",
        "QuantizedCache",
        "QuantizedCacheConfig",
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

    # Data

    # Feature Extractor

    # Generation

    # Integrations

    # Model Cards

    # TF 2.0 <=> PyTorch conversion utilities
    from .models import *

    # Hindi Causal LM imports
    # Pipelines
    # Tokenization
    # Trainer
    # Files and general utilities
    from .utils import (
        is_flax_available,
        is_sentencepiece_available,
        is_tf_available,
        is_timm_available,
        is_tokenizers_available,
        is_torch_available,
        is_torchvision_available,
        is_vision_available,
        logging,
    )

    # bitsandbytes config

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_tokenizers_objects import *
    else:
        pass

    try:
        if not (is_sentencepiece_available() and is_tokenizers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_sentencepiece_and_tokenizers_objects import *
    else:
        pass

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_vision_objects import *
    else:
        pass

    try:
        if not is_torchvision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torchvision_objects import *
    else:
        pass

    try:
        if not (is_torchvision_available() and is_timm_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_timm_and_torchvision_objects import *
    else:
        pass

    # Modeling
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_pt_objects import *
    else:
        # Debugging

        # Optimization

        # Trainer
        pass

    # TensorFlow
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # Import the same objects as dummies to get them in the namespace.
        # They will raise an import error if the user tries to instantiate / use them.
        from .utils.dummy_tf_objects import *
    else:

        # Optimization
        pass

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # Import the same objects as dummies to get them in the namespace.
        # They will raise an import error if the user tries to instantiate / use them.
        from .utils.dummy_flax_objects import *
    else:
        pass

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
