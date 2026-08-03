# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from transformers import VoxCPM2Model
from transformers.models.voxcpm2.convert_voxcpm2_weights_to_hf import convert_voxcpm2_state_dict
from transformers.testing_utils import require_torch

from .test_modeling_voxcpm2 import get_tiny_voxcpm2_config


@require_torch
def test_checkpoint_key_conversion_loads_strictly():
    source_model = VoxCPM2Model(get_tiny_voxcpm2_config())
    model_state_dict = {}
    audio_vae_state_dict = {}
    for key, value in source_model.state_dict().items():
        source_key = key.replace("parametrizations.weight.original0", "weight_g")
        source_key = source_key.replace("parametrizations.weight.original1", "weight_v")
        if source_key.startswith("audio_vae."):
            audio_vae_state_dict[source_key.removeprefix("audio_vae.")] = value
        else:
            model_state_dict[source_key] = value

    converted_state_dict = convert_voxcpm2_state_dict(model_state_dict, audio_vae_state_dict)

    assert set(converted_state_dict) == set(source_model.state_dict())
    converted_model = VoxCPM2Model(get_tiny_voxcpm2_config())
    incompatible_keys = converted_model.load_state_dict(converted_state_dict, strict=True)
    assert not incompatible_keys.missing_keys
    assert not incompatible_keys.unexpected_keys
