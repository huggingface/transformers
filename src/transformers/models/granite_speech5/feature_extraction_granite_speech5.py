# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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
"""Backwards-compatibility shim: re-exports the legacy ``GraniteSpeech5FeatureExtractor`` name as
a deprecated alias of [`GraniteSpeech5AudioProcessor`]. Importing or instantiating the alias emits
a ``FutureWarning``; the alias is removed in transformers v5.15 (see ADR 0002).
"""

from ...audio_processing_base import make_legacy_audio_processor_alias
from .audio_processing_granite_speech5 import GraniteSpeech5AudioProcessor


GraniteSpeech5FeatureExtractor = make_legacy_audio_processor_alias(
    GraniteSpeech5AudioProcessor, "GraniteSpeech5FeatureExtractor"
)


__all__ = ["GraniteSpeech5FeatureExtractor"]
