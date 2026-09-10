# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import FrozenInstanceError, replace

import pytest

from transformers.audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig
from transformers.image_utils import SizeDict


@pytest.mark.parametrize("config", [SizeDict(height=80), StftConfig(), MelScaleConfig(), SpectrogramConfig()])
def test_dictionary_access_and_round_trip(config):
    serialized = config.to_dict()
    assert dict(config) == serialized
    assert config == serialized
    assert serialized == config
    assert type(config).from_dict(json.loads(json.dumps(serialized))) == config
    for key, value in serialized.items():
        assert key in config
        assert config.get(key) == value
    assert config.get("missing", 42) == 42
    assert "to_dict" not in config
    with pytest.raises(KeyError):
        config["to_dict"]
    with pytest.raises(KeyError):
        config["missing"]


def test_size_dict_compatibility():
    size = SizeDict(height=80)
    assert size["width"] is None
    assert "width" not in size
    assert size.get("width", 42) == 42
    assert size | {"width": 40} == SizeDict(height=80, width=40)
    assert size | SizeDict(width=40) == SizeDict(height=80, width=40)
    assert {"height": 20, "width": 40} | size == {"height": 80, "width": 40}
    assert isinstance({} | size, dict)
    assert hash(size) == hash(SizeDict(height=80))
    size["height"] = 100
    assert size.height == 100
    with pytest.raises(KeyError):
        size["to_dict"] = 1


def test_nested_audio_configs():
    source = {
        "stft_config": {"n_fft": 512, "wkwargs": {"custom_window_arg": None}},
        "mel_scale_config": {"n_mels": 80, "norm": "slaney"},
    }
    config = SpectrogramConfig.from_dict(source)
    assert isinstance(config.stft_config, StftConfig)
    assert isinstance(config.mel_scale_config, MelScaleConfig)
    assert isinstance(source["stft_config"], dict)
    assert config.stft_config.wkwargs == {"custom_window_arg": None}
    assert SpectrogramConfig.from_dict(config.to_dict()) == config
    assert SpectrogramConfig.from_dict({"stft_config": config.stft_config}).stft_config is config.stft_config
    assert (config | {"mel_scale_config": None}).mel_scale_config is None
    merged = config | {"stft_config": {"hop_length": 100}}
    assert isinstance(merged.stft_config, StftConfig)
    assert merged.stft_config.hop_length == 100
    assert merged.stft_config.n_fft == 400  # Union replaces the entire nested field.
    assert config.stft_config.n_fft == 512
    assert config != config.stft_config


@pytest.mark.parametrize("config", [StftConfig(), MelScaleConfig(), SpectrogramConfig()])
def test_audio_configs_stay_frozen(config):
    key = next(iter(dict(config)))
    with pytest.raises(FrozenInstanceError):
        setattr(config, key, config[key])
    with pytest.raises(TypeError):
        config[key] = config[key]
    assert replace(config) == config
    assert hash(replace(config)) == hash(config)


@pytest.mark.parametrize("cls", [SizeDict, StftConfig, MelScaleConfig, SpectrogramConfig])
def test_unknown_config_fields_are_rejected(cls):
    with pytest.raises(TypeError):
        cls.from_dict({"misspelled_field": 80})
    with pytest.raises(TypeError):
        SpectrogramConfig.from_dict({"stft_config": {"n_ftt": 512}})
