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

import json

from transformers import AutoProcessor, GenerationConfig, VoxCPM2Model
from transformers.models.voxcpm2.convert_voxcpm2_weights_to_hf import convert_checkpoint, convert_voxcpm2_state_dict
from transformers.testing_utils import require_torch

from .test_modeling_voxcpm2 import get_tiny_voxcpm2_config
from .test_processing_voxcpm2 import get_tiny_voxcpm2_processor


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


@require_torch
def test_checkpoint_conversion_saves_native_artifacts(tmp_path):
    import torch
    from safetensors.torch import save_file

    source_path = tmp_path / "source"
    output_path = tmp_path / "converted"
    source_path.mkdir()
    processor = get_tiny_voxcpm2_processor()
    config = get_tiny_voxcpm2_config()
    config.audio_start_token_id = processor.audio_start_token_id
    config.reference_audio_start_token_id = processor.reference_audio_start_token_id
    config.reference_audio_end_token_id = processor.reference_audio_end_token_id
    source_model = VoxCPM2Model(config)
    config.save_pretrained(source_path)
    processor.tokenizer.save_pretrained(source_path)

    model_state_dict = {}
    audio_vae_state_dict = {}
    for key, value in source_model.state_dict().items():
        source_key = key.replace("parametrizations.weight.original0", "weight_g")
        source_key = source_key.replace("parametrizations.weight.original1", "weight_v")
        value = value.detach().cpu().contiguous().clone()
        if source_key.startswith("audio_vae."):
            audio_vae_state_dict[source_key.removeprefix("audio_vae.")] = value
        else:
            model_state_dict[source_key] = value
    save_file(model_state_dict, source_path / "model.safetensors")
    torch.save({"state_dict": audio_vae_state_dict}, source_path / "audiovae.pth")

    converted_model, converted_processor = convert_checkpoint(source_path, output_path)

    assert not any(parameter.is_meta for parameter in converted_model.parameters())
    reloaded_model = VoxCPM2Model.from_pretrained(output_path)
    reloaded_processor = AutoProcessor.from_pretrained(output_path)
    generation_config = GenerationConfig.from_pretrained(output_path)
    assert type(reloaded_processor) is type(converted_processor)
    assert generation_config.min_new_tokens == 4
    assert generation_config.max_new_tokens == 2000
    assert generation_config.guidance_scale == 2.0
    with open(output_path / "config.json", encoding="utf-8") as config_file:
        saved_config = json.load(config_file)
    with open(output_path / "tokenizer_config.json", encoding="utf-8") as tokenizer_file:
        saved_tokenizer = json.load(tokenizer_file)
    assert saved_config["architectures"] == ["VoxCPM2Model"]
    assert "auto_map" not in saved_tokenizer
    for key, value in source_model.state_dict().items():
        torch.testing.assert_close(reloaded_model.state_dict()[key], value, rtol=0, atol=0)
