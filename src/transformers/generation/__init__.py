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

from ..utils import OptionalDependencyNotAvailable, _LazyModule, is_flax_available, is_tf_available, is_torch_available


_import_structure = {
    "configuration_utils": ["GenerationConfig"],
    "streamers": ["TextIteratorStreamer", "TextStreamer"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["logits_process"] = [
        "EpsilonLogitsWarper",
        "EtaLogitsWarper",
        "ForcedBOSTokenLogitsProcessor",
        "ForcedEOSTokenLogitsProcessor",
        "HammingDiversityLogitsProcessor",
        "InfNanRemoveLogitsProcessor",
        "LogitsProcessor",
        "LogitsProcessorList",
        "LogitsWarper",
        "MinLengthLogitsProcessor",
        "MinNewTokensLengthLogitsProcessor",
        "NoBadWordsLogitsProcessor",
        "NoRepeatNGramLogitsProcessor",
        "PrefixConstrainedLogitsProcessor",
        "RepetitionPenaltyLogitsProcessor",
        "SequenceBiasLogitsProcessor",
        "EncoderRepetitionPenaltyLogitsProcessor",
        "TemperatureLogitsWarper",
        "TopKLogitsWarper",
        "TopPLogitsWarper",
        "TypicalLogitsWarper",
        "EncoderNoRepeatNGramLogitsProcessor",
        "ExponentialDecayLengthPenalty",
        "LogitNormalization",
    ]
    _import_structure["stopping_criteria"] = [
        "MaxNewTokensCriteria",
        "MaxLengthCriteria",
        "MaxTimeCriteria",
        "StoppingCriteria",
        "StoppingCriteriaList",
        "validate_stopping_criteria",
    ]
    _import_structure["utils"] = [
        "GenerationMixin",
        "top_k_top_p_filtering",
    ]
    _import_structure["strategies.utils"] = [
        "GreedySearchEncoderDecoderOutput",
        "GreedySearchDecoderOnlyOutput",
        "SampleEncoderDecoderOutput",
        "SampleDecoderOnlyOutput",
        "BeamSearchEncoderDecoderOutput",
        "BeamSearchDecoderOnlyOutput",
        "BeamSampleEncoderDecoderOutput",
        "BeamSampleDecoderOnlyOutput",
        "ContrastiveSearchEncoderDecoderOutput",
        "ContrastiveSearchDecoderOnlyOutput",
        "GenerationStrategy",
    ]
    _import_structure["strategies.beam_utils"] = [
        "BeamHypotheses",
        "BeamScorer",
        "BeamSearchScorer",
    ]
    _import_structure["strategies.beam_constrained_decoder"] = [
        "Constraint",
        "ConstraintListState",
        "DisjunctiveConstraint",
        "PhrasalConstraint",
        "ConstrainedBeamSearchScorer",
        "ConstrainedDecoder"
    ]
    _import_structure["strategies.beam_search_decoder"] = [
        "BeamSearchDecoder",
    ]
    _import_structure["strategies.beam_sampling_decoder"] = [
        "BeamSamplingDecoder",
    ]
    _import_structure["strategies.diverse_group_beam_decoder"] = [
        "DiverseGroupBeamDecoder",
    ]
    _import_structure["strategies.contrastive_decoder"] = [
        "ContrastiveDecoder",
    ]
    _import_structure["strategies.greedy_decoder"] = [
        "GreedyDecoder",
    ]
    _import_structure["strategies.sampling_decoder"] = [
        "SamplingDecoder",
    ]
    _import_structure["assisted_decoder"] = [
        "AssistedDecoder",
    ]



try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tf_logits_process"] = [
        "TFForcedBOSTokenLogitsProcessor",
        "TFForcedEOSTokenLogitsProcessor",
        "TFLogitsProcessor",
        "TFLogitsProcessorList",
        "TFLogitsWarper",
        "TFMinLengthLogitsProcessor",
        "TFNoBadWordsLogitsProcessor",
        "TFNoRepeatNGramLogitsProcessor",
        "TFRepetitionPenaltyLogitsProcessor",
        "TFTemperatureLogitsWarper",
        "TFTopKLogitsWarper",
        "TFTopPLogitsWarper",
        "TFForceTokensLogitsProcessor",
        "TFSuppressTokensAtBeginLogitsProcessor",
        "TFSuppressTokensLogitsProcessor",
    ]
    _import_structure["tf_utils"] = [
        "TFGenerationMixin",
        "tf_top_k_top_p_filtering",
        "TFGreedySearchDecoderOnlyOutput",
        "TFGreedySearchEncoderDecoderOutput",
        "TFSampleEncoderDecoderOutput",
        "TFSampleDecoderOnlyOutput",
        "TFBeamSearchEncoderDecoderOutput",
        "TFBeamSearchDecoderOnlyOutput",
        "TFBeamSampleEncoderDecoderOutput",
        "TFBeamSampleDecoderOnlyOutput",
        "TFContrastiveSearchEncoderDecoderOutput",
        "TFContrastiveSearchDecoderOnlyOutput",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["flax_logits_process"] = [
        "FlaxForcedBOSTokenLogitsProcessor",
        "FlaxForcedEOSTokenLogitsProcessor",
        "FlaxLogitsProcessor",
        "FlaxLogitsProcessorList",
        "FlaxLogitsWarper",
        "FlaxMinLengthLogitsProcessor",
        "FlaxTemperatureLogitsWarper",
        "FlaxTopKLogitsWarper",
        "FlaxTopPLogitsWarper",
    ]
    _import_structure["flax_utils"] = [
        "FlaxGenerationMixin",
        "FlaxGreedySearchOutput",
        "FlaxSampleOutput",
        "FlaxBeamSearchOutput",
    ]

if TYPE_CHECKING:
    from .configuration_utils import GenerationConfig
    from .streamers import TextIteratorStreamer, TextStreamer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .strategies.beam_constrained_decoder import Constraint, ConstraintListState, \
    DisjunctiveConstraint, PhrasalConstraint, ConstrainedBeamSearchScorer
        from .strategies.beam_utils import BeamHypotheses, BeamScorer, BeamSearchScorer
        from .logits_process import (
            EncoderNoRepeatNGramLogitsProcessor,
            EncoderRepetitionPenaltyLogitsProcessor,
            EpsilonLogitsWarper,
            EtaLogitsWarper,
            ExponentialDecayLengthPenalty,
            ForcedBOSTokenLogitsProcessor,
            ForcedEOSTokenLogitsProcessor,
            HammingDiversityLogitsProcessor,
            InfNanRemoveLogitsProcessor,
            LogitNormalization,
            LogitsProcessor,
            LogitsProcessorList,
            LogitsWarper,
            MinLengthLogitsProcessor,
            MinNewTokensLengthLogitsProcessor,
            NoBadWordsLogitsProcessor,
            NoRepeatNGramLogitsProcessor,
            PrefixConstrainedLogitsProcessor,
            RepetitionPenaltyLogitsProcessor,
            SequenceBiasLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
            TypicalLogitsWarper,
        )
        from .stopping_criteria import (
            MaxLengthCriteria,
            MaxNewTokensCriteria,
            MaxTimeCriteria,
            StoppingCriteria,
            StoppingCriteriaList,
            validate_stopping_criteria,
        )
        from .utils import (
            BeamSampleDecoderOnlyOutput,
            BeamSampleEncoderDecoderOutput,
            ContrastiveSearchDecoderOnlyOutput,
            ContrastiveSearchEncoderDecoderOutput,
            GenerationMixin,
            GreedySearchDecoderOnlyOutput,
            GreedySearchEncoderDecoderOutput,
            SampleDecoderOnlyOutput,
            SampleEncoderDecoderOutput,
            top_k_top_p_filtering,
        )
        from .strategies.utils import BeamSearchDecoderOnlyOutput, BeamSearchEncoderDecoderOutput

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tf_logits_process import (
            TFForcedBOSTokenLogitsProcessor,
            TFForcedEOSTokenLogitsProcessor,
            TFForceTokensLogitsProcessor,
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
        from .tf_utils import (
            TFBeamSampleDecoderOnlyOutput,
            TFBeamSampleEncoderDecoderOutput,
            TFBeamSearchDecoderOnlyOutput,
            TFBeamSearchEncoderDecoderOutput,
            TFContrastiveSearchDecoderOnlyOutput,
            TFContrastiveSearchEncoderDecoderOutput,
            TFGenerationMixin,
            TFGreedySearchDecoderOnlyOutput,
            TFGreedySearchEncoderDecoderOutput,
            TFSampleDecoderOnlyOutput,
            TFSampleEncoderDecoderOutput,
            tf_top_k_top_p_filtering,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .flax_logits_process import (
            FlaxForcedBOSTokenLogitsProcessor,
            FlaxForcedEOSTokenLogitsProcessor,
            FlaxLogitsProcessor,
            FlaxLogitsProcessorList,
            FlaxLogitsWarper,
            FlaxMinLengthLogitsProcessor,
            FlaxTemperatureLogitsWarper,
            FlaxTopKLogitsWarper,
            FlaxTopPLogitsWarper,
        )
        from .flax_utils import FlaxBeamSearchOutput, FlaxGenerationMixin, FlaxGreedySearchOutput, FlaxSampleOutput
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
