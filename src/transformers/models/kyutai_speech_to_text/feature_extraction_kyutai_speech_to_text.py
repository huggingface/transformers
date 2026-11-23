# Copyright 2025 Kyutai and The HuggingFace Inc. team. All rights reserved.
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
"""Backwards-compatibility shim: re-exports the legacy ``KyutaiSpeechToTextFeatureExtractor``
name as a deprecated alias of [`KyutaiSpeechToTextAudioProcessor`]. Importing or instantiating
the alias emits a ``FutureWarning``; the alias is removed in transformers v5.15 (see ADR 0002).
"""

from ...audio_processing_base import make_legacy_audio_processor_alias
from .audio_processing_kyutai_speech_to_text import KyutaiSpeechToTextAudioProcessor


KyutaiSpeechToTextFeatureExtractor = make_legacy_audio_processor_alias(
    KyutaiSpeechToTextAudioProcessor, "KyutaiSpeechToTextFeatureExtractor"
)


__all__ = ["KyutaiSpeechToTextFeatureExtractor"]
