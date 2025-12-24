# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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
from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from ...activations import ACT2FN
from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, is_torch_available, logging
from ...utils.generic import check_model_inputs
from ..audioflamingo3.modeling_audioflamingo3 import (
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3MultiModalProjector,
    AudioFlamingo3PreTrainedModel,
)
from ..audioflamingo3.processing_audioflamingo3 import AudioFlamingo3Processor, AudioFlamingo3ProcessorKwargs
from ..glm.modeling_glm import GlmRotaryEmbedding
from ..llama.modeling_llama import LlamaAttention, eager_attention_forward, rotate_half
from .configuration_glmasr import GlmAsrConfig, GlmAsrEncoderConfig


logger = logging.get_logger(__name__)


class GlmAsrProcessorKwargs(AudioFlamingo3ProcessorKwargs): ...


class GlmAsrProcessor(AudioFlamingo3Processor):
    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
        audio_token="<|pad|>",
        default_transcription_prompt="Please transcribe this audio into text",
        max_audio_len=600,  # TODO : ???
    ):
        super().__init__(
            feature_extractor,
            tokenizer,
            chat_template=chat_template,
            audio_token=audio_token,
            default_transcription_prompt=default_transcription_prompt,
            max_audio_len=max_audio_len,
        )

    def _get_audio_token_length(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        merge_factor = 4
        for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
            audio_lengths = (audio_lengths + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        num_tokens = (audio_lengths - merge_factor) // merge_factor + 1
        return num_tokens

    def apply_transcription_request(
        self,
        audio: Union[str, list[str], AudioInput],
        prompt: Optional[Union[str, list[str]]] = None,
        **kwargs: Unpack[GlmAsrProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare inputs for automatic speech recognition without manually writing the default transcription prompt.

        Args:
            audio (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                Audio to transcribe. Strings are interpreted as local paths or URLs and will be loaded automatically by
                the chat template loader; NumPy arrays and PyTorch tensors are forwarded directly.
            prompt (`str` or `list[str]`, *optional*):
                Custom prompt(s) to include in the user turn. A list must be the same length as the batch. When `None`,
                each sample uses `"Transcribe the input speech."`.
            **kwargs:
                Additional keyword arguments forwarded to [`~AudioFlamingo3Processor.apply_chat_template`] (for example
                `text_kwargs`, `audio_kwargs`, ...).

        Returns:
            [`BatchFeature`]: Processor outputs ready to be passed to [`AudioFlamingo3ForConditionalGeneration.generate`].

        """

        if isinstance(audio, str):
            audio_items: list[Union[str, np.ndarray]] = [audio]
        elif isinstance(audio, (list, tuple)) and audio and all(isinstance(el, str) for el in audio):
            audio_items = list(audio)
        else:
            audio_items = list(make_list_of_audio(audio))
            if is_torch_available():
                audio_items = [el.detach().cpu().numpy() if isinstance(el, torch.Tensor) else el for el in audio_items]

        batch_size = len(audio_items)
        if batch_size == 0:
            raise ValueError("`audio` must contain at least one sample.")

        if prompt is None:
            prompts = [self.default_transcription_prompt] * batch_size
        elif isinstance(prompt, str):
            prompts = [prompt] * batch_size
        elif isinstance(prompt, (list, tuple)):
            if len(prompt) != batch_size:
                raise ValueError(
                    f"Received {len(prompt)} prompt(s) for {batch_size} audio sample(s); counts must match."
                )
            prompts = []
            for item in prompt:
                if item is None:
                    prompts.append(self.default_transcription_prompt)
                elif isinstance(item, str):
                    prompts.append(item)
                else:
                    raise TypeError("Each prompt must be a string or `None`.")
        else:
            raise TypeError("`prompt` must be a string, a sequence of strings, or `None`.")

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": audio_item}
                        if isinstance(audio_item, str)
                        else {"type": "audio", "audio": audio_item},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            for prompt_text, audio_item in zip(prompts, audio_items)
        ]

        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            **kwargs,
        )


class GlmAsrRotaryEmbedding(GlmRotaryEmbedding): ...


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class GlmAsrAttention(LlamaAttention):
    def __init__(self, config: GlmAsrConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = False
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GlmAsrMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class GlmAsrEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GlmAsrConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GlmAsrAttention(config=config, layer_idx=layer_idx)

        self.mlp = GlmAsrMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class GlmAsrPreTrainedModel(AudioFlamingo3PreTrainedModel): ...


# TODO: @eustlb, this is what WhisperEncoder should look like
class GlmAsrEncoder(GlmAsrPreTrainedModel):
    config: GlmAsrEncoderConfig
    main_input_name = "input_features"
    input_modalities = "audio"
    _no_split_modules = ["GlmAsrEncoderLayer"]

    def __init__(self, config: GlmAsrEncoderConfig):
        super().__init__(config)
        self.conv1 = nn.Conv1d(config.num_mel_bins, config.hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList(
            [GlmAsrEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.rotary_emb = GlmAsrRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(self, input_features, **kwargs: Unpack[TransformersKwargs]):
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.transpose(1, 2)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(
            hidden_states, position_ids=torch.arange(hidden_states.shape[1], device=hidden_states.device)[None, :]
        )

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, position_embeddings=position_embeddings, **kwargs)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class GlmAsrMultiModalProjector(AudioFlamingo3MultiModalProjector):
    def __init__(self, config: GlmAsrConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.audio_config.intermediate_size, config.text_config.hidden_size * 2)
        self.linear_2 = nn.Linear(config.text_config.hidden_size * 2, config.text_config.hidden_size)


class GlmAsrForConditionalGeneration(AudioFlamingo3ForConditionalGeneration):
    def get_audio_features(
        self, input_features: torch.FloatTensor, input_features_mask: torch.Tensor
    ) -> torch.FloatTensor:
        audio_outputs = self.audio_tower(input_features)
        audio_hidden_states = audio_outputs.last_hidden_state
        audio_hidden_states = audio_hidden_states.reshape(
            input_features.shape[0], -1, self.config.audio_config.intermediate_size
        )
        audio_embeds = self.multi_modal_projector(audio_hidden_states)

        audio_lengths = input_features_mask.sum(-1)
        for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
            audio_lengths = (audio_lengths + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        merge_factor = 4
        post_lengths = (audio_lengths - merge_factor) // merge_factor + 1

        valid_mask = torch.arange(audio_embeds.shape[1], device=post_lengths.device)[None, :] < post_lengths[:, None]
        audio_embeds = audio_embeds[valid_mask.to(audio_embeds.device)]
        return audio_embeds


__all__ = ["GlmAsrEncoder", "GlmAsrForConditionalGeneration", "GlmAsrProcessor", "GlmAsrPreTrainedModel"]
