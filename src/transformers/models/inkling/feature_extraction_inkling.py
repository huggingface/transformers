# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Backwards-compatibility shim: re-exports the legacy ``InklingFeatureExtractor`` name as a
deprecated alias of [`InklingAudioProcessor`]. Importing or instantiating the alias emits a
``FutureWarning``; the alias is removed in transformers v5.15 (see ADR 0002).
"""

from ...audio_processing_base import make_legacy_audio_processor_alias
from .audio_processing_inkling import InklingAudioProcessor


InklingFeatureExtractor = make_legacy_audio_processor_alias(InklingAudioProcessor, "InklingFeatureExtractor")


__all__ = ["InklingFeatureExtractor"]
