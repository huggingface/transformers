# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from ..utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available


_import_structure = {
    "configuration_utils": [
        "BaseWatermarkingConfig",
        "CompileConfig",
        "GenerationConfig",
        "GenerationMode",
        "SynthIDTextWatermarkingConfig",
        "WatermarkingConfig",
    ],
    "streamers": ["AsyncTextIteratorStreamer", "BaseStreamer", "TextIteratorStreamer", "TextStreamer"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["candidate_generator"] = [
        "AssistedCandidateGenerator",
        "CandidateGenerator",
        "EarlyExitCandidateGenerator",
        "PromptLookupCandidateGenerator",
    ]
    _import_structure["logits_process"] = [
        "AlternatingCodebooksLogitsProcessor",
        "ClassifierFreeGuidanceLogitsProcessor",
        "EncoderNoRepeatNGramLogitsProcessor",
        "EncoderRepetitionPenaltyLogitsProcessor",
        "EpsilonLogitsWarper",
        "EtaLogitsWarper",
        "ExponentialDecayLengthPenalty",
        "ForcedBOSTokenLogitsProcessor",
        "ForcedEOSTokenLogitsProcessor",
        "InfNanRemoveLogitsProcessor",
        "LogitNormalization",
        "LogitsProcessor",
        "LogitsProcessorList",
        "MinLengthLogitsProcessor",
        "MinNewTokensLengthLogitsProcessor",
        "MinPLogitsWarper",
        "NoBadWordsLogitsProcessor",
        "NoRepeatNGramLogitsProcessor",
        "PrefixConstrainedLogitsProcessor",
        "RepetitionPenaltyLogitsProcessor",
        "SequenceBiasLogitsProcessor",
        "SuppressTokensLogitsProcessor",
        "SuppressTokensAtBeginLogitsProcessor",
        "SynthIDTextWatermarkLogitsProcessor",
        "TemperatureLogitsWarper",
        "TopHLogitsWarper",
        "TopKLogitsWarper",
        "TopPLogitsWarper",
        "TypicalLogitsWarper",
        "UnbatchedClassifierFreeGuidanceLogitsProcessor",
        "WhisperTimeStampLogitsProcessor",
        "WatermarkLogitsProcessor",
    ]
    _import_structure["stopping_criteria"] = [
        "MaxLengthCriteria",
        "MaxTimeCriteria",
        "ConfidenceCriteria",
        "EosTokenCriteria",
        "StoppingCriteria",
        "StoppingCriteriaList",
        "validate_stopping_criteria",
        "StopStringCriteria",
    ]
    _import_structure["continuous_batching"] = [
        "ContinuousMixin",
    ]
    _import_structure["utils"] = [
        "GenerationMixin",
        "GenerateBeamDecoderOnlyOutput",
        "GenerateBeamEncoderDecoderOutput",
        "GenerateDecoderOnlyOutput",
        "GenerateEncoderDecoderOutput",
    ]
    _import_structure["watermarking"] = [
        "WatermarkDetector",
        "WatermarkDetectorOutput",
        "BayesianDetectorModel",
        "BayesianDetectorConfig",
        "SynthIDTextWatermarkDetector",
    ]


if TYPE_CHECKING:
    from .configuration_utils import (
        BaseWatermarkingConfig,
        CompileConfig,
        GenerationConfig,
        GenerationMode,
        SynthIDTextWatermarkingConfig,
        WatermarkingConfig,
    )
    from .streamers import AsyncTextIteratorStreamer, BaseStreamer, TextIteratorStreamer, TextStreamer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .candidate_generator import (
            AssistedCandidateGenerator,
            CandidateGenerator,
            EarlyExitCandidateGenerator,
            PromptLookupCandidateGenerator,
        )
        from .continuous_batching import ContinuousMixin
        from .logits_process import (
            AlternatingCodebooksLogitsProcessor,
            ClassifierFreeGuidanceLogitsProcessor,
            EncoderNoRepeatNGramLogitsProcessor,
            EncoderRepetitionPenaltyLogitsProcessor,
            EpsilonLogitsWarper,
            EtaLogitsWarper,
            ExponentialDecayLengthPenalty,
            ForcedBOSTokenLogitsProcessor,
            ForcedEOSTokenLogitsProcessor,
            InfNanRemoveLogitsProcessor,
            LogitNormalization,
            LogitsProcessor,
            LogitsProcessorList,
            MinLengthLogitsProcessor,
            MinNewTokensLengthLogitsProcessor,
            MinPLogitsWarper,
            NoBadWordsLogitsProcessor,
            NoRepeatNGramLogitsProcessor,
            PrefixConstrainedLogitsProcessor,
            RepetitionPenaltyLogitsProcessor,
            SequenceBiasLogitsProcessor,
            SuppressTokensAtBeginLogitsProcessor,
            SuppressTokensLogitsProcessor,
            SynthIDTextWatermarkLogitsProcessor,
            TemperatureLogitsWarper,
            TopHLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
            TypicalLogitsWarper,
            UnbatchedClassifierFreeGuidanceLogitsProcessor,
            WatermarkLogitsProcessor,
            WhisperTimeStampLogitsProcessor,
        )
        from .stopping_criteria import (
            ConfidenceCriteria,
            EosTokenCriteria,
            MaxLengthCriteria,
            MaxTimeCriteria,
            StoppingCriteria,
            StoppingCriteriaList,
            StopStringCriteria,
            validate_stopping_criteria,
        )
        from .utils import (
            GenerateBeamDecoderOnlyOutput,
            GenerateBeamEncoderDecoderOutput,
            GenerateDecoderOnlyOutput,
            GenerateEncoderDecoderOutput,
            GenerationMixin,
        )
        from .watermarking import (
            BayesianDetectorConfig,
            BayesianDetectorModel,
            SynthIDTextWatermarkDetector,
            WatermarkDetector,
            WatermarkDetectorOutput,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
