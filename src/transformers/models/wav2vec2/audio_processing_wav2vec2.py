# Copyright 2025 The HuggingFace Inc. team.
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

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import NormalizationConfig


class Wav2Vec2AudioProcessor(TorchAudioBackend):
    model_input_names = ["input_values", "attention_mask"]

    sample_rate = 16000
    force_mono = True
    do_values_normalize = True
    normalization_config = NormalizationConfig(method="zero_mean_unit_var")


__all__ = ["Wav2Vec2AudioProcessor"]
