# Copyright 2026 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Backwards-compatibility shim: re-exports the legacy ``Xcodec2FeatureExtractor`` name as a
deprecated alias of [`Xcodec2AudioProcessor`]. Importing or instantiating the alias emits a
``FutureWarning``; the alias is removed in transformers v5.15 (see ADR 0002).
"""

from ...audio_processing_base import make_legacy_audio_processor_alias
from .audio_processing_xcodec2 import Xcodec2AudioProcessor


Xcodec2FeatureExtractor = make_legacy_audio_processor_alias(Xcodec2AudioProcessor, "Xcodec2FeatureExtractor")


__all__ = ["Xcodec2FeatureExtractor"]
