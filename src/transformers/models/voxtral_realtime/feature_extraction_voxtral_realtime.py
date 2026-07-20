# Copyright 2026 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Backwards-compatibility shim: re-exports the legacy ``VoxtralRealtimeFeatureExtractor`` name as a
deprecated alias of [`VoxtralRealtimeAudioProcessor`]. Importing or instantiating the alias emits a
``FutureWarning``; the alias is removed in transformers v5.15 (see ADR 0002).
"""

from ...audio_processing_base import make_legacy_audio_processor_alias
from .audio_processing_voxtral_realtime import VoxtralRealtimeAudioProcessor


VoxtralRealtimeFeatureExtractor = make_legacy_audio_processor_alias(VoxtralRealtimeAudioProcessor, "VoxtralRealtimeFeatureExtractor")


__all__ = ["VoxtralRealtimeFeatureExtractor"]
