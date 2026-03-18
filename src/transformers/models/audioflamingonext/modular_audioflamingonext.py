# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig
from ..musicflamingo.configuration_musicflamingo import MusicFlamingoConfig
from ..musicflamingo.modeling_musicflamingo import MusicFlamingoForConditionalGeneration, MusicFlamingoPreTrainedModel
from ..musicflamingo.processing_musicflamingo import MusicFlamingoProcessor, MusicFlamingoProcessorKwargs


@auto_docstring(checkpoint="nvidia/audio-flamingo-next")
@strict(accept_kwargs=True)
class AudioFlamingoNextConfig(MusicFlamingoConfig):
    model_type = "audioflamingonext"
    sub_configs = {"audio_config": AutoConfig, "text_config": AutoConfig}
    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 1200}
        self.max_position_embeddings = self.rope_parameters["rope_theta"]

        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get("model_type", "audioflamingo3_encoder")
            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["audioflamingo3_encoder"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()

        PreTrainedConfig.__post_init__(self, **kwargs)


class AudioFlamingoNextProcessorKwargs(MusicFlamingoProcessorKwargs): ...


class AudioFlamingoNextProcessor(MusicFlamingoProcessor):
    pass


class AudioFlamingoNextPreTrainedModel(MusicFlamingoPreTrainedModel):
    pass


@auto_docstring(
    custom_intro="""
    The AudioFlamingoNext model which is architecturally identical to MusicFlamingo.
    """
)
class AudioFlamingoNextForConditionalGeneration(MusicFlamingoForConditionalGeneration):
    pass


__all__ = [
    "AudioFlamingoNextConfig",
    "AudioFlamingoNextProcessor",
    "AudioFlamingoNextPreTrainedModel",
    "AudioFlamingoNextForConditionalGeneration",
]
