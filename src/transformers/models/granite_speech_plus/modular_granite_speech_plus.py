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
"""Granite Speech Plus model, a Granite Speech variant whose projector consumes the concatenation of the
encoder's final hidden states with an arbitrary subset of its intermediate hidden states."""

from collections.abc import Container

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...modeling_outputs import BaseModelOutputWithPooling
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..granite_speech.configuration_granite_speech import GraniteSpeechConfig, GraniteSpeechEncoderConfig
from ..granite_speech.modeling_granite_speech import (
    GraniteSpeechCausalLMOutputWithPast,
    GraniteSpeechConformerAttention,
    GraniteSpeechConformerBlock,
    GraniteSpeechConformerConvModule,
    GraniteSpeechConformerDepthWiseConv1d,
    GraniteSpeechConformerFeedForward,
    GraniteSpeechCTCEncoder,
    GraniteSpeechEncoderProjector,
    GraniteSpeechForConditionalGeneration,
    GraniteSpeechPreTrainedModel,
)


@auto_docstring(checkpoint="ibm-granite/granite-speech-4.1-2b-plus")
@strict
class GraniteSpeechPlusEncoderConfig(GraniteSpeechEncoderConfig):
    r"""
    cat_hidden_layers (`list[int]`, *optional*):
        Indices of encoder conformer layers whose outputs are concatenated with the final encoder
        output (along the feature dimension) before being passed to the projector. When set, the
        projector's ``encoder_hidden_size`` must equal
        ``encoder_config.hidden_dim * (len(cat_hidden_layers) + 1)``.

    Example:

    ```python
    >>> from transformers import GraniteSpeechPlusEncoderConfig, GraniteSpeechPlusCTCEncoder

    >>> # Initializing a GraniteSpeechPlusEncoderConfig
    >>> configuration = GraniteSpeechPlusEncoderConfig()

    >>> # Initializing a GraniteSpeechPlusCTCEncoder (with random weights)
    >>> model = GraniteSpeechPlusCTCEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    cat_hidden_layers: list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.cat_hidden_layers is not None:
            for idx in self.cat_hidden_layers:
                if idx < 0 or idx >= self.num_layers:
                    raise ValueError(
                        f"cat_hidden_layers index {idx} is out of range [0, {self.num_layers})."
                    )


@auto_docstring(checkpoint="ibm-granite/granite-speech-4.1-2b-plus")
@strict
class GraniteSpeechPlusConfig(GraniteSpeechConfig):
    r"""
    projector_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Blip2QFormerConfig`):
        The config object or dictionary of the audio projector.
    has_lora_adapter (`bool`, *optional*, defaults to `True`):
        Indicates whether or not the model has a lora adapter that should only
        be activate when processing audio inputs.
    downsample_rate (`int`, *optional*, defaults to 5):
        Downsample rate for the audio feature extractor.
    window_size (`int`, *optional*, defaults to 15):
        Window size for the audio feature projector.
    encoder_hidden_layers (`list[int]`, *optional*):
        Indices of encoder conformer layers whose outputs are concatenated with the final encoder
        output (along the feature dimension) before being passed to the projector. When set, the
        projector's ``encoder_hidden_size`` must equal
        ``encoder_config.hidden_dim * (len(encoder_hidden_layers) + 1)``.

    Example:

    ```python
    >>> from transformers import GraniteSpeechPlusConfig, GraniteSpeechPlusForConditionalGeneration

    >>> # Initializing a GraniteSpeechPlusConfig
    >>> configuration = GraniteSpeechPlusConfig()

    >>> # Initializing a GraniteSpeechPlusForConditionalGeneration (with random weights)
    >>> model = GraniteSpeechPlusForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""


    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        if self.encoder_config.cat_hidden_layers is not None:
            num_concat = len(self.encoder_config.cat_hidden_layers) + 1
            if self.projector_config.encoder_hidden_size != self.encoder_config.hidden_dim * num_concat:
                raise ValueError(
                    f"projector encoder_hidden_size {self.projector_config.encoder_hidden_size} "
                    f"must equal encoder hidden_dim * {num_concat} = "
                    f"{self.encoder_config.hidden_dim * num_concat}."
                )




class GraniteSpeechPlusCTCEncoder(GraniteSpeechCTCEncoder):
    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        hidden_states: torch.Tensor,
        returned_hidden_states: Container[int] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        hidden_states = self.input_linear(hidden_states)
        exported_hidden_states = []
        if returned_hidden_states is None:
            returned_hidden_states = []
        if 0 in returned_hidden_states:
            exported_hidden_states.append(hidden_states)
        for idx, layer in enumerate(self.layers, start=1):
            hidden_states = layer(hidden_states, attention_dists=self.attention_dists)
            if idx in returned_hidden_states:
                exported_hidden_states.append(hidden_states)

            if idx == self.num_layers // 2:
                hidden_states_mid = hidden_states.clone()
                hidden_states_mid = self.out(hidden_states_mid)
                hidden_states += self.out_mid(nn.Softmax(dim=-1)(hidden_states_mid))
        if len(exported_hidden_states) > 0:
            hidden_states = torch.cat(exported_hidden_states + [hidden_states], dim=-1)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


@auto_docstring(
    custom_intro="""
    The Granite Speech Plus model, a Granite Speech variant whose projector consumes the concatenation of the
    encoder's final hidden states with an arbitrary subset of its intermediate hidden states.
    """
)
class GraniteSpeechPlusForConditionalGeneration(GraniteSpeechForConditionalGeneration):
    @can_return_tuple
    @auto_docstring
    def get_audio_features(
        self, input_features: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        audio_outputs = self.encoder(
            input_features, returned_hidden_states=self.config.encoder_hidden_layers, return_dict=True, **kwargs
        )
        encoder_embeds = audio_outputs.last_hidden_state
        projected_embeds = self.projector(encoder_embeds)
        audio_outputs.pooler_output = projected_embeds

        return audio_outputs


__all__ = [
    "GraniteSpeechPlusConfig",
    "GraniteSpeechPlusEncoderConfig",
    "GraniteSpeechPlusCTCEncoder",
    "GraniteSpeechPlusForConditionalGeneration",
    "GraniteSpeechPlusPreTrainedModel",
]
