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

# When adding a new object to this init, remember to add it twice: once inside the `_import_structure` dictionary and
# once inside the `if TYPE_CHECKING` branch. The `TYPE_CHECKING` should have import statements as usual, but they are
# only there for type checking. The actual implementation is in the `else` branch.

# Configurations
from .configuration_utils import PretrainedConfig
from .models.auto.configuration_auto import CONFIG_MAPPING, MODEL_NAMES_MAPPING, AutoConfig
from .models.auto.tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer

# Import all these explicitly to make them available in the package namespace.
from .models.auto.auto_factory import (
    # Models
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
    MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING,
    MODEL_FOR_AUDIO_XVECTOR_MAPPING,
    MODEL_FOR_BACKBONE_MAPPING,
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_CTC_MAPPING,
    MODEL_FOR_DEPTH_ESTIMATION_MAPPING,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_IMAGE_MAPPING,
    MODEL_FOR_IMAGE_SEGMENTATION_MAPPING,
    MODEL_FOR_IMAGE_TO_IMAGE_MAPPING,
    MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING,
    MODEL_FOR_KEYPOINT_DETECTION_MAPPING,
    MODEL_FOR_MASK_GENERATION_MAPPING,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    MODEL_FOR_OBJECT_DETECTION_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_TEXT_ENCODING_MAPPING,
    MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING,
    MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING,
    MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING,
    MODEL_FOR_TIME_SERIES_PREDICTION_MAPPING,
    MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING,
    MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
    MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING,
    MODEL_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    # Processors
    FEATURE_EXTRACTOR_MAPPING,
    IMAGE_PROCESSOR_MAPPING,
    PROCESSOR_MAPPING,
    TOKENIZER_MAPPING,
    VISION_PROCESSOR_MAPPING,
    # Factory classes
    AutoBackbone,
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForAudioFrameClassification,
    AutoModelForAudioXVector,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForDepthEstimation,
    AutoModelForDocumentQuestionAnswering,
    AutoModelForImageClassification,
    AutoModelForImageSegmentation,
    AutoModelForImageToImage,
    AutoModelForInstanceSegmentation,
    AutoModelForKeypointDetection,
    AutoModelForMaskGeneration,
    AutoModelForMaskedImageModeling,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForNextSentencePrediction,
    AutoModelForObjectDetection,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSemanticSegmentation,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoModelForTableQuestionAnswering,
    AutoModelForTextEncoding,
    AutoModelForTextToSpectrogram,
    AutoModelForTextToWaveform,
    AutoModelForTimeSeriesClassification,
    AutoModelForTimeSeriesPrediction,
    AutoModelForTimeSeriesRegression,
    AutoModelForTokenClassification,
    AutoModelForUniversalSegmentation,
    AutoModelForVideoClassification,
    AutoModelForVision2Seq,
    AutoModelForVisualQuestionAnswering,
    AutoModelForZeroShotImageClassification,
    AutoModelForZeroShotObjectDetection,
    AutoModelWithLMHead,
    AutoProcessor,
    AutoTokenizer,
    AutoVisionProcessor,
)

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
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    cached_file,
    cached_path,
    copy_func,
    default_data_collator,
    download_url,
    extract_commit_hash,
    find_labels,
    get_full_repo_name,
    has_file,
    is_apex_available,
    is_bio_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_decord_available,
    is_detectron2_available,
    is_essentia_available,
    is_faiss_available,
    is_flax_available,
    is_g2p_en_available,
    is_in_notebook,
    is_ipex_available,
    is_keras_nlp_available,
    is_librosa_available,
    is_natten_available,
    is_ninja_available,
    is_onnx_available,
    is_pandas_available,
    is_phonemizer_available,
    is_pretty_midi_available,
    is_pydub_available,
    is_pypi_package_name_matching,
    is_pyctcdecode_available,
    is_pytesseract_available,
    is_pytest_running,
    is_rjieba_available,
    is_safetensors_available,
    is_scatter_available,
    is_scipy_available,
    is_sentencepiece_available,
    is_sklearn_available,
    is_speech_available,
    is_sudachi_available,
    is_tensorflow_probability_available,
    is_tensorflow_text_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tensorrt_fx_available,
    is_torch_tpu_available,
    is_torchvision_available,
    is_torchaudio_available,
    is_vision_available,
    logging,
)

# Dynamic modules
_import_structure = {
    "adapter_config": [],
    "benchmark": [],
    "commands": [],
    "data": [],
    "deepspeed": [],
    "dependency_versions_check": [],
    "generation": ["GenerationConfig", "TextIteratorStreamer", "TextStreamer"],
    "image_transforms": [],
    "modelcard": [],
    "quantizers": [],
    "tokenization_utils": [],
    "tokenization_utils_base": [],
}

# Direct imports for type-checking
if TYPE_CHECKING:
    # Configuration
    from .configuration_utils import PretrainedConfig

    # Models
    from .models import *

    from .models.hindi_causal_lm import (
        HindiCausalLMConfig,  # Just include the config during type checking
    )
    
    # Import modeling classes only if torch is available during type checking
    if is_torch_available():
        from .models.hindi_causal_lm import (
            HindiCausalLMForCausalLM,
            HindiCausalLMModel,
            HindiCausalLMPreTrainedModel,
        )
        
    # Import tokenizer if sentencepiece is available
    if is_sentencepiece_available():
        from .models.hindi_causal_lm import HindiCausalLMTokenizer

    # Tokenizers
    from .tokenization_utils import PreTrainedTokenizer
    from .tokenization_utils_base import (
        AddedToken,
        BatchEncoding,
        CharSpan,
        PreTrainedTokenizerBase,
        SpecialTokensMixin,
        TokenSpan,
    )

    # Data
    from .data import DataProcessor

    # Feature Extractor
    from .feature_extraction_utils import FeatureExtractionMixin
    from .image_processing_utils import ImageProcessingMixin

    # Other
    from .utils import logging

# Lazy loading for modules that require one or more optional dependencies
else:
    import importlib
    import os
    import sys

    class _LazyModule(_LazyModule):
        """
        Module class that surfaces all objects but only performs associated imports when the objects are requested.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        def _get_module(self, module_name):
            return importlib.import_module("." + module_name, self.__name__)

    # Define the structure used for lazy loading
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)

    # Import models lazily
    from .models import *

    # Add config to the model import structure
    _import_structure["models.hindi_causal_lm"] = ["HindiCausalLMConfig"]

    # Add model classes only if torch is available
    if is_torch_available():
        _import_structure["models.hindi_causal_lm"].extend([
            "HindiCausalLMModel",
            "HindiCausalLMForCausalLM",
            "HindiCausalLMPreTrainedModel",
        ])

    # Add tokenizer only if sentencepiece is available
    if is_sentencepiece_available():
        _import_structure["models.hindi_causal_lm"].append("HindiCausalLMTokenizer")

# Import version from package setup
from .__version__ import __version__
