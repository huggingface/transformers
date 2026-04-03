# Copyright 2025 The HuggingFace Inc. team and Google LLC. All rights reserved.
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
from .audio_processing_lasr import LasrAudioProcessor


LasrFeatureExtractor = deprecated_feature_extractor(LasrAudioProcessor, "LasrFeatureExtractor")

__all__ = ["LasrFeatureExtractor"]
