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

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...modeling_outputs import BaseModelOutputWithPooling
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..granite_speech.configuration_granite_speech import GraniteSpeechConfig, GraniteSpeechEncoderConfig
from ..granite_speech.modeling_granite_speech import (
    GraniteSpeechCTCEncoder,
    GraniteSpeechForConditionalGeneration,
    GraniteSpeechPreTrainedModel,
)


@auto_docstring(checkpoint="ibm-granite/granite-speech-4.1-2b-plus")
@strict
class GraniteSpeechPlusEncoderConfig(GraniteSpeechEncoderConfig):
    r"""
    feedforward_mult (`int`, *optional*, defaults to 4):
        Multiplier for the up/down projections in the encoder's feedforward layers;
        The projections will have intermediate dim of size `hidden_dim * feedforward_mult`.
    output_dim (`int`, *optional*, defaults to 42):
        Intermediate dimension of the feedforward projections in the conformer
        to be added to every other encoder block's output.
    context_size (`int`, *optional*, defaults to 200):
        Context size to be used in conformer attention.
    max_pos_emb (`int`, *optional*, defaults to 512):
        Max pos embeds to be used in attention (shaw's relative positional encoding).
    conv_expansion_factor (`int`, *optional*, defaults to 2):
        Intermediate dimension to be used in conformer convolutions.
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
            for idx in self.encoder_config.cat_hidden_layers:
                if idx < 0 or idx >= self.encoder_config.num_layers:
                    raise ValueError(
                        f"cat_hidden_layers index {idx} is out of range [0, {self.encoder_config.num_layers})."
                    )
        if self.encoder_config.cat_hidden_layers is not None:
            num_concat = len(self.encoder_config.cat_hidden_layers) + 1
            if self.projector_config.encoder_hidden_size != self.encoder_config.hidden_dim * num_concat:
                raise ValueError(
                    f"projector encoder_hidden_size {self.projector_config.encoder_hidden_size} "
                    f"must equal encoder hidden_dim * {num_concat} = "
                    f"{self.encoder_config.hidden_dim * num_concat}."
                )


class GraniteSpeechPlusPreTrainedModel(GraniteSpeechPreTrainedModel): ...


class GraniteSpeechPlusCTCEncoder(GraniteSpeechCTCEncoder):
    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        hidden_states = self.input_linear(hidden_states)
        cat_layers = set(self.config.cat_hidden_layers or [])
        exported_hidden_states = []

        if 0 in cat_layers:
            exported_hidden_states.append(hidden_states)

        for idx, layer in enumerate(self.layers, start=1):
            hidden_states = layer(hidden_states, attention_dists=self.attention_dists)

            if idx in cat_layers:
                exported_hidden_states.append(hidden_states)

            if idx == self.num_layers // 2:
                hidden_states_mid = hidden_states.clone()
                hidden_states_mid = self.out(hidden_states_mid)
                hidden_states += self.out_mid(nn.Softmax(dim=-1)(hidden_states_mid))

        if exported_hidden_states:
            hidden_states = torch.cat([*exported_hidden_states, hidden_states], dim=-1)

        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


@auto_docstring(
    custom_intro="""
    The Granite Speech Plus model, a Granite Speech variant whose projector consumes the concatenation of the
    encoder's final hidden states with an arbitrary subset of its intermediate hidden states.
    """
)
class GraniteSpeechPlusForConditionalGeneration(GraniteSpeechForConditionalGeneration): ...


__all__ = [
    "GraniteSpeechPlusConfig",
    "GraniteSpeechPlusEncoderConfig",
    "GraniteSpeechPlusCTCEncoder",
    "GraniteSpeechPlusForConditionalGeneration",
    "GraniteSpeechPlusPreTrainedModel",
]
