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
from typing import TYPE_CHECKING

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .auto_factory import get_values
    from .configuration_auto import CONFIG_MAPPING, MODEL_NAMES_MAPPING, AutoConfig
    from .feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING, AutoFeatureExtractor
    from .image_processing_auto import IMAGE_PROCESSOR_MAPPING, AutoImageProcessor
    from .modeling_auto import (
        MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
        MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING,
        MODEL_FOR_AUDIO_TOKENIZATION_MAPPING,
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
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
        MODEL_FOR_IMAGE_TO_IMAGE_MAPPING,
        MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING,
        MODEL_FOR_KEYPOINT_DETECTION_MAPPING,
        MODEL_FOR_KEYPOINT_MATCHING_MAPPING,
        MODEL_FOR_MASK_GENERATION_MAPPING,
        MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
        MODEL_FOR_MASKED_LM_MAPPING,
        MODEL_FOR_MULTIMODAL_LM_MAPPING,
        MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
        MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
        MODEL_FOR_OBJECT_DETECTION_MAPPING,
        MODEL_FOR_PRETRAINING_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_RETRIEVAL_MAPPING,
        MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
        MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_TABLE_RECOGNITION_MAPPING,
        MODEL_FOR_TEXT_ENCODING_MAPPING,
        MODEL_FOR_TEXT_RECOGNITION_MAPPING,
        MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING,
        MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING,
        MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING,
        MODEL_FOR_TIME_SERIES_PREDICTION_MAPPING,
        MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING,
        MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING,
        MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,
        MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING,
        MODEL_MAPPING,
        AutoBackbone,
        AutoModel,
        AutoModelForAudioClassification,
        AutoModelForAudioFrameClassification,
        AutoModelForAudioTokenization,
        AutoModelForAudioXVector,
        AutoModelForCausalLM,
        AutoModelForCTC,
        AutoModelForDepthEstimation,
        AutoModelForDocumentQuestionAnswering,
        AutoModelForImageClassification,
        AutoModelForImageSegmentation,
        AutoModelForImageTextToText,
        AutoModelForImageToImage,
        AutoModelForInstanceSegmentation,
        AutoModelForKeypointDetection,
        AutoModelForKeypointMatching,
        AutoModelForMaskedImageModeling,
        AutoModelForMaskedLM,
        AutoModelForMaskGeneration,
        AutoModelForMultimodalLM,
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
        AutoModelForTableRecognition,
        AutoModelForTextEncoding,
        AutoModelForTextRecognition,
        AutoModelForTextToSpectrogram,
        AutoModelForTextToWaveform,
        AutoModelForTimeSeriesPrediction,
        AutoModelForTokenClassification,
        AutoModelForUniversalSegmentation,
        AutoModelForVideoClassification,
        AutoModelForVisualQuestionAnswering,
        AutoModelForZeroShotImageClassification,
        AutoModelForZeroShotObjectDetection,
    )
    from .processing_auto import PROCESSOR_MAPPING, AutoProcessor
    from .tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
    from .video_processing_auto import VIDEO_PROCESSOR_MAPPING, AutoVideoProcessor
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
