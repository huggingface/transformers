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

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_bidirectional_mask, create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3Model,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..vibevoice.modeling_vibevoice import VibeVoiceDiffusionHeadSinusoidalEmbedding
from .configuration_yue2 import YuE2Config
from .generation_yue2 import YuE2GenerationMixin


@auto_docstring(custom_intro="""Outputs of `YuE2ForConditionalGeneration`.""")
@dataclass
class YuE2OutputWithPast(CausalLMOutputWithPast):
    r"""
    loss (`torch.FloatTensor` of shape `()`, *optional*, returned when `labels` is passed):
        Language modeling loss of the autoregressive tokens.
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`, *optional*):
        Prediction scores of the language modeling head, when the inputs are tokens.
    velocity (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`, *optional*):
        Flow-matching velocity predicted for each acoustic latent frame, when the inputs are `audio_latents`.
    """

    velocity: torch.FloatTensor | None = None


class YuE2RMSNorm(Qwen3RMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The original's arithmetic: the inverse root mean square is computed in float32 but applied in the input dtype.
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + self.variance_epsilon).to(hidden_states.dtype) * self.weight


class YuE2MLP(Qwen3MLP):
    pass


class YuE2RotaryEmbedding(Qwen3RotaryEmbedding):
    pass


class YuE2Attention(Qwen3Attention):
    def __init__(self, config: YuE2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.layer_type
        del self.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class YuE2DecoderLayer(Qwen3DecoderLayer):
    """
    A mixture-of-transformers layer: one set of weights (attention, MLP and norms) for the autoregressive tokens and
    another, the `nar_*` modules, for the acoustic latents generated by flow matching. The latents attend to the tokens
    before them, and to each other in both directions.
    """

    def __init__(self, config: YuE2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.nar_input_layernorm = YuE2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.nar_self_attn = YuE2Attention(config=config, layer_idx=layer_idx)
        self.nar_self_attn.is_causal = False
        self.nar_pre_mlp_layernorm = YuE2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.nar_mlp = YuE2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        is_latent: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if is_latent:
            input_layernorm, self_attn = self.nar_input_layernorm, self.nar_self_attn
            post_attention_layernorm, mlp = self.nar_pre_mlp_layernorm, self.nar_mlp
        else:
            input_layernorm, self_attn = self.input_layernorm, self.self_attn
            post_attention_layernorm, mlp = self.post_attention_layernorm, self.mlp

        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)
        hidden_states, _ = self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = post_attention_layernorm(hidden_states)
        hidden_states = mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class YuE2TimestepSinusoidalEmbedding(VibeVoiceDiffusionHeadSinusoidalEmbedding):
    @staticmethod
    def compute_default_freq(config: YuE2Config) -> torch.Tensor:
        dim = config.frequency_embedding_size // 2
        return torch.exp(-math.log(config.max_period) * torch.arange(dim, dtype=torch.float32) / dim)


class YuE2TimestepEmbedding(nn.Module):
    """Sinusoidal embedding of the flow-matching time followed by a two-layer MLP."""

    def __init__(self, config: YuE2Config):
        super().__init__()
        self.sinusoidal_embedding = YuE2TimestepSinusoidalEmbedding(config)
        self.fc1 = nn.Linear(config.frequency_embedding_size, config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        embedding = self.sinusoidal_embedding(timesteps.reshape(-1)).to(self.fc1.weight.dtype)
        embedding = self.fc2(self.act_fn(self.fc1(embedding)))
        return embedding.reshape(*timesteps.shape, -1)


class YuE2LatentPositionEmbedding(nn.Module):
    """
    Fixed sinusoidal embedding of the position of an acoustic latent frame (interleaved sines and cosines). The released
    checkpoint stores the table, whose values differ from `compute_default_embedding` by a bfloat16 rounding for 0.1% of
    them, so it is loaded rather than computed.
    """

    def __init__(self, config: YuE2Config):
        super().__init__()
        self.embedding = nn.Buffer(self.compute_default_embedding(config))

    @staticmethod
    def compute_default_embedding(config: YuE2Config) -> torch.Tensor:
        positions = torch.arange(config.max_latent_frames, dtype=torch.float32)[:, None]
        div_term = torch.exp(
            torch.arange(0, config.hidden_size, 2, dtype=torch.float32)
            * (-math.log(config.max_period) / config.hidden_size)
        )
        embedding = torch.zeros(config.max_latent_frames, config.hidden_size)
        embedding[:, 0::2] = torch.sin(positions * div_term)
        embedding[:, 1::2] = torch.cos(positions * div_term)
        return embedding

    def forward(self, position_ids: torch.LongTensor) -> torch.Tensor:
        return self.embedding[position_ids]


@auto_docstring
class YuE2PreTrainedModel(Qwen3PreTrainedModel):
    input_modalities = ("text", "audio")

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, YuE2TimestepSinusoidalEmbedding):
            init.copy_(module.freq, module.compute_default_freq(self.config))
        elif isinstance(module, YuE2LatentPositionEmbedding):
            init.copy_(module.embedding, module.compute_default_embedding(self.config))


@auto_docstring(
    custom_intro="""
    The YuE2 backbone: a mixture-of-transformers decoder over the prompt, the symbolic score and the semantic tokens
    (autoregressive), and over the acoustic latents of the song (flow matching), returning the normalized hidden states.
    """
)
class YuE2Model(Qwen3Model):
    def __init__(self, config: YuE2Config):
        super().__init__(config)
        # no padding index: the padding token is also the beginning of every prompt
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.latent_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.timestep_embedding = YuE2TimestepEmbedding(config)
        self.latent_position_embedding = YuE2LatentPositionEmbedding(config)
        del self.has_sliding_layers
        del self.padding_idx
        del self.vocab_size

    def embed_audio_latents(self, audio_latents: torch.FloatTensor, timesteps: torch.Tensor) -> torch.FloatTensor:
        """
        Embeds noisy acoustic latents at flow-matching time `timesteps`. The latent positions open and close with a frame
        of zeros (the original's latent start and end positions), so `num_frames` latents fill `num_frames + 2`
        positions.
        """
        latents = nn.functional.pad(audio_latents, (0, 0, 1, 1)).to(self.latent_proj.weight.dtype)
        hidden_states = self.latent_proj(latents)
        # embedded at every position, as the original: one row per position in its MLP's matmuls, whose rounding in
        # half precision depends on the number of rows
        timesteps = timesteps[:, None].expand(-1, latents.shape[1])
        hidden_states = hidden_states + self.timestep_embedding(timesteps)
        position_ids = torch.arange(latents.shape[1], device=latents.device)
        return hidden_states + self.latent_position_embedding(position_ids)[None].to(hidden_states.dtype)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        audio_latents: torch.FloatTensor | None = None,
        timesteps: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        audio_latents (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`, *optional*):
            Noisy acoustic latents, in place of `input_ids`: the layers then run their `nar_*` weights, with the latents
            attending to `past_key_values` (the cache of the tokens that condition them) and to each other in both
            directions. Two positions of zeros frame the latents, so the outputs hold `num_frames + 2` positions.
        timesteps (`torch.Tensor` of shape `(batch_size,)`, *optional*):
            Flow-matching time of `audio_latents`, from 1 (noise) to 0 (data), after the shift of the time schedule.
        """
        is_latent = audio_latents is not None
        if is_latent:
            if input_ids is not None or inputs_embeds is not None:
                raise ValueError("Pass either `audio_latents` or the tokens (`input_ids` or `inputs_embeds`).")
            if timesteps is None:
                raise ValueError("`audio_latents` need their flow-matching `timesteps`.")
            inputs_embeds = self.embed_audio_latents(audio_latents, timesteps)
        elif (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            if attention_mask is not None and attention_mask.ndim == 2:
                # positions skip the left padding, as `generate` derives them
                position_ids = attention_mask.long().cumsum(-1)[:, -inputs_embeds.shape[1] :] - 1
            else:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
                position_ids = position_ids.unsqueeze(0)

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }
        if is_latent:
            # the latents see the whole cached prefix and each other
            attention_mask = create_bidirectional_mask(**mask_kwargs)
        elif not isinstance(attention_mask, dict):
            attention_mask = create_causal_mask(**mask_kwargs, position_ids=position_ids)
        else:
            # it may already have been prepared by `generate`
            attention_mask = attention_mask["full_attention"]

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                is_latent=is_latent,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring(
    custom_intro="""
    The YuE2 song generation model: from lyrics and a style prompt, it writes a symbolic score (ABC notation) and the
    semantic tokens of the song autoregressively, then generates the acoustic latents of each semantic token by flow
    matching. The processor decodes the latents into 48 kHz stereo audio with the acoustic tokenizer.
    """
)
class YuE2ForConditionalGeneration(YuE2PreTrainedModel, YuE2GenerationMixin):
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: YuE2Config):
        super().__init__(config)
        self.model = YuE2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.velocity_head = nn.Linear(config.hidden_size, config.latent_dim)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        audio_latents: torch.FloatTensor | None = None,
        timesteps: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> YuE2OutputWithPast:
        r"""
        audio_latents (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`, *optional*):
            Noisy acoustic latents, in place of `input_ids`: the model then predicts their flow-matching `velocity`,
            conditioned on the tokens cached in `past_key_values`. Each `audio_latents` call appends its positions to the
            cache; crop it back to the tokens before the next call.
        timesteps (`torch.Tensor` of shape `(batch_size,)`, *optional*):
            Flow-matching time of `audio_latents`, from 1 (noise) to 0 (data), after the shift of the time schedule.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels of the autoregressive tokens for the language modeling loss. Indices should either be in
            `[0, ..., config.vocab_size]` or -100 (masked).
        """
        if labels is not None and audio_latents is not None:
            raise ValueError("`labels` are token labels: the loss is only computed on the autoregressive tokens.")

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            audio_latents=audio_latents,
            timesteps=timesteps,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        logits = velocity = loss = None
        if audio_latents is not None:
            # the first and last positions frame the latents
            velocity = self.velocity_head(hidden_states[:, 1:-1])
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return YuE2OutputWithPast(
            loss=loss,
            logits=logits,
            velocity=velocity,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_audio_latent_velocity(
        self, audio_latents: torch.FloatTensor, timesteps: torch.Tensor, past_key_values: Cache
    ) -> torch.FloatTensor:
        """
        Flow-matching velocity of `audio_latents` conditioned on the cached tokens, leaving the cache as it was.

        Args:
            audio_latents (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`):
                Noisy acoustic latents.
            timesteps (`torch.Tensor` of shape `(batch_size,)`):
                Their flow-matching time, after the shift of the time schedule.
            past_key_values (`Cache`):
                The cache of the tokens that condition the latents.
        """
        num_cached_tokens = past_key_values.get_seq_length()
        velocity = self(audio_latents=audio_latents, timesteps=timesteps, past_key_values=past_key_values).velocity
        # drop the latent positions this call appended
        past_key_values.crop(num_cached_tokens - past_key_values.get_seq_length())
        return velocity


__all__ = ["YuE2PreTrainedModel", "YuE2Model", "YuE2ForConditionalGeneration"]
