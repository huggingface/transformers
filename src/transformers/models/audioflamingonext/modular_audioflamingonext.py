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

from ...utils import auto_docstring
from ..musicflamingo.configuration_musicflamingo import MusicFlamingoConfig
from ..musicflamingo.modeling_musicflamingo import MusicFlamingoForConditionalGeneration, MusicFlamingoPreTrainedModel
from ..musicflamingo.processing_musicflamingo import MusicFlamingoProcessor, MusicFlamingoProcessorKwargs


@auto_docstring(checkpoint="nvidia/audio-flamingo-next-hf")
@strict(accept_kwargs=True)
class AudioFlamingoNextConfig(MusicFlamingoConfig):
    model_type = "audioflamingonext"

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get("model_type", "audioflamingo3_encoder")
        elif self.audio_config is None:
            self.audio_config = {"model_type": "audioflamingo3_encoder"}
        super().__post_init__(**kwargs)


class AudioFlamingoNextProcessorKwargs(MusicFlamingoProcessorKwargs): ...


class AudioFlamingoNextProcessor(MusicFlamingoProcessor):
    r"""
    Constructs an AudioFlamingoNext processor which wraps an AudioFlamingoNext feature extractor and an AudioFlamingoNext
    tokenizer into a single processor.

    [`AudioFlamingoNextProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`] and
    [`Qwen2TokenizerFast`]. See the [`~AudioFlamingoNextProcessor.__call__`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`]):
            The feature extractor is a required input.
        tokenizer ([`Qwen2TokenizerFast`]):
            The tokenizer is a required input.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the tokenizer's default chat
            template will be used.
        audio_token (`Optional[str]`, *optional*, defaults to `"<sound>"`):
            Special token used to represent audio inputs in the chat template.
        audio_bos_token (`Optional[str]`, *optional*, defaults to `"<|sound_bos|>"`):
            Special token used to represent the beginning of audio.
        audio_eos_token (`Optional[str]`, *optional*, defaults to `"<|sound_eos|>"`):
            Special token used to represent the end of audio.
        max_audio_len (`int`, *optional*, defaults to 1800):
            Maximum length of audio sequences in seconds. Audio longer than this will be truncated.
    """

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
        audio_token="<sound>",
        audio_bos_token="<|sound_bos|>",
        audio_eos_token="<|sound_eos|>",
        max_audio_len=1800,
    ):
        super().__init__(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            audio_token=audio_token,
            audio_bos_token=audio_bos_token,
            audio_eos_token=audio_eos_token,
            max_audio_len=max_audio_len,
        )


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
