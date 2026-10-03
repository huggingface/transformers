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

from collections.abc import Callable

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..clip.modeling_clip import CLIPMLP, CLIPEncoderLayer
from ..glmasr.configuration_glmasr import GlmAsrEncoderConfig
from ..glmasr.modeling_glmasr import (
    GlmAsrAttention,
    GlmAsrRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..pe_audio.modeling_pe_audio import PeAudioPreTrainedModel


@auto_docstring(checkpoint="nvidia/Nemotron-3-Diarization")
@strict
class NemotronSpeechEncoderConfig(GlmAsrEncoderConfig):
    r"""
    subsampling_factor (`int`, *optional*, defaults to 8):
        Number of consecutive spectrogram frames stacked into one encoder frame.
    use_qk_norm (`bool`, *optional*, defaults to `False`):
        Whether to apply a layer norm to the queries and keys of each attention head, before the rotary embedding.

    Example:

    ```python
    >>> from transformers import NemotronSpeechEncoder, NemotronSpeechEncoderConfig

    >>> # Initializing a NemotronSpeechEncoder configuration
    >>> configuration = NemotronSpeechEncoderConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = NemotronSpeechEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "nemotron_speech_encoder"
    base_config_key = "audio_config"

    hidden_size: int = 512
    num_hidden_layers: int = 31
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    subsampling_factor: int = 8
    max_position_embeddings: int = 5000
    use_qk_norm: bool = False

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        kwargs.setdefault("partial_rotary_factor", 1.0)
        PreTrainedConfig.__post_init__(self, **kwargs)

    def validate_architecture(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size` ({self.hidden_size}) must be divisible by `num_attention_heads` "
                f"({self.num_attention_heads})."
            )


class NemotronSpeechEncoderFeatureStacking(nn.Module):
    """Stacks `subsampling_factor` consecutive spectrogram frames and projects them to the encoder hidden size."""

    def __init__(self, config: NemotronSpeechEncoderConfig):
        super().__init__()
        self.subsampling_factor = config.subsampling_factor
        self.projection = nn.Linear(config.subsampling_factor * config.num_mel_bins, config.hidden_size, bias=False)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, num_mel_bins = input_features.shape
        # The original zero-pads the last incomplete group of frames.
        padding = -num_frames % self.subsampling_factor
        input_features = nn.functional.pad(input_features, (0, 0, 0, padding))
        stacked = input_features.reshape(
            batch_size, (num_frames + padding) // self.subsampling_factor, num_mel_bins * self.subsampling_factor
        )
        return self.projection(stacked)


class NemotronSpeechEncoderRotaryEmbedding(GlmAsrRotaryEmbedding): ...


class NemotronSpeechEncoderAttention(GlmAsrAttention):
    def __init__(self, config: NemotronSpeechEncoderConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # The original fused query/key/value projection has no bias while the output projection has one.
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        # CODEPATH: the Nemotron-3.5-Transcribe ASR encoder normalizes queries and keys, Nemotron-3-Diarization does not.
        self.q_norm = nn.LayerNorm(self.head_dim) if config.use_qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if config.use_qk_norm else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class NemotronSpeechEncoderMLP(CLIPMLP): ...


class NemotronSpeechEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config: NemotronSpeechEncoderConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = NemotronSpeechEncoderAttention(config, layer_idx)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)


@auto_docstring
class NemotronSpeechEncoderPreTrainedModel(PeAudioPreTrainedModel):
    config: NemotronSpeechEncoderConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    input_modalities = "audio"
    _no_split_modules = ["NemotronSpeechEncoderLayer"]
    _can_record_outputs = {
        "hidden_states": NemotronSpeechEncoderLayer,
        "attentions": NemotronSpeechEncoderAttention,
    }

    def _init_weights(self, module):
        raise AttributeError("Not needed")


@auto_docstring(
    custom_intro="""
    The Nemotron speech encoder: stacks spectrogram frames and encodes them with a pre-norm Transformer using rotary
    position embeddings. It is the audio tower of Nemotron speech models such as Nemotron-3-Diarization.
    """
)
class NemotronSpeechEncoder(NemotronSpeechEncoderPreTrainedModel):
    def __init__(self, config: NemotronSpeechEncoderConfig):
        super().__init__(config)
        self.embedder = NemotronSpeechEncoderFeatureStacking(config)
        self.input_layer_norm = nn.LayerNorm(config.hidden_size)
        self.layers = nn.ModuleList(
            [NemotronSpeechEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.rotary_emb = NemotronSpeechEncoderRotaryEmbedding(config)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        if (input_features is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of `input_features` and `inputs_embeds`.")

        if inputs_embeds is None:
            if attention_mask is not None:
                input_features = input_features.masked_fill(~attention_mask[..., None].bool(), 0.0)
            inputs_embeds = self.embedder(input_features)
            # if inputs_embeds is provided, we expect attention_mask already downsampled
            if attention_mask is not None:
                attention_mask = attention_mask[:, :: self.config.subsampling_factor].bool()

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)[None, :]
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        attention_mask = create_bidirectional_mask(
            config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask
        )
        hidden_states = self.input_layer_norm(inputs_embeds)
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return BaseModelOutput(last_hidden_state=self.layer_norm(hidden_states))


__all__ = [
    "NemotronSpeechEncoderConfig",
    "NemotronSpeechEncoder",
    "NemotronSpeechEncoderPreTrainedModel",
]
