# Copyright 2026 The HuggingFace Inc. team.
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

from ...audio_processing_backends import NumpyAudioBackend
from .audio_processing_gemma4 import Gemma4AudioProcessorMixin


class Gemma4AudioProcessorNumpy(Gemma4AudioProcessorMixin, NumpyAudioBackend):
    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        # Zero the padded frames, as the legacy extractor does.
        mask = output.get("audio_features_mask")
        if mask is not None and "audio_features" in output:
            features = output["audio_features"]
            output["audio_features"] = features * mask.astype(features.dtype)[..., None]
        return output


__all__ = ["Gemma4AudioProcessorNumpy"]
