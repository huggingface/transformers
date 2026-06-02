# Copyright 2023 The HuggingFace Inc. team.
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
from ...utils.deprecation import deprecated_feature_extractor
from .audio_processing_seamless_m4t import SeamlessM4tAudioProcessor


SeamlessM4TFeatureExtractor = deprecated_feature_extractor(SeamlessM4tAudioProcessor, "SeamlessM4TFeatureExtractor")

__all__ = ["SeamlessM4TFeatureExtractor"]
