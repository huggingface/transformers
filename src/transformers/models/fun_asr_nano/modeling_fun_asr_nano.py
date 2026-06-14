# Copyright 2025 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Fun-ASR-Nano model."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ..auto import AutoModelForCausalLM
from .configuration_fun_asr_nano import (
    FunAsrNanoAdaptorConfig,
    FunAsrNanoConfig,
    FunAsrNanoCtcConfig,
    FunAsrNanoEncoderConfig,
)


logger = logging.get_logger(__name__)


@dataclass
class FunAsrNanoCausalLMOutput(ModelOutput):
    """Output of the Fun-ASR-Nano model for conditional generation."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: tuple | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None


# ============================================================================
# Audio Encoder Components (SANM - Self-Attention with FSMN Memory)
# Key names match original checkpoint: self_attn.linear_q_k_v, self_attn.linear_out,
# self_attn.fsmn_block, feed_forward.w_1, feed_forward.w_2, norm1, norm2
# ============================================================================


class FunAsrNanoSinusoidalPositionEncoder(nn.Module):
    """Sinusoidal positional encoding generated on the fly."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device, dtype=x.dtype).unsqueeze(0)

        log_timescale_increment = math.log(10000.0) / (input_dim / 2 - 1)
        inv_timescales = torch.exp(
            torch.arange(0, input_dim // 2, device=x.device, dtype=x.dtype) * (-log_timescale_increment)
        )
        scaled_time = positions.unsqueeze(2) * inv_timescales.unsqueeze(0).unsqueeze(0)
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)

        return x + encoding


class FunAsrNanoSANMAttention(nn.Module):
    """Self-Attention with FSMN Memory (SANM).

    State dict keys:
        self_attn.linear_q_k_v.{weight,bias}
        self_attn.linear_out.{weight,bias}
        self_attn.fsmn_block.weight  (Conv1d depthwise, no bias)
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        num_heads: int,
        attention_dropout_rate: float = 0.0,
        kernel_size: int = 11,
        sanm_shift: int = 0,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.d_k = hidden_size // num_heads
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.linear_q_k_v = nn.Linear(in_features, hidden_size * 3)
        self.linear_out = nn.Linear(hidden_size, hidden_size)

        # FSMN depthwise conv (key: self_attn.fsmn_block.weight)
        self.fsmn_block = nn.Conv1d(
            hidden_size, hidden_size, kernel_size, stride=1, padding=0, groups=hidden_size, bias=False
        )
        left_padding = (kernel_size - 1) // 2
        if sanm_shift > 0:
            left_padding = left_padding + sanm_shift
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

        self.dropout = nn.Dropout(p=attention_dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.size()

        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, self.hidden_size, dim=-1)

        # FSMN memory path
        if mask is not None:
            mask_expanded = mask.view(b, -1, 1)
            v_masked = v * mask_expanded
        else:
            v_masked = v

        fsmn_out = v_masked.transpose(1, 2)
        fsmn_out = self.pad_fn(fsmn_out)
        fsmn_out = self.fsmn_block(fsmn_out)
        fsmn_out = fsmn_out.transpose(1, 2)
        fsmn_memory = fsmn_out + v_masked
        fsmn_memory = self.dropout(fsmn_memory)
        if mask is not None:
            fsmn_memory = fsmn_memory * mask_expanded

        # Multi-head attention path
        q = q.view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        v_heads = v.view(b, t, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.d_k**-0.5)

        if mask is not None:
            mask_for_attn = mask.unsqueeze(1).eq(0)
            scores = scores.masked_fill(mask_for_attn, float("-inf"))
            attn_weights = torch.softmax(scores, dim=-1).masked_fill(mask_for_attn, 0.0)
        else:
            attn_weights = torch.softmax(scores, dim=-1)

        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v_heads)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, self.hidden_size)
        attn_output = self.linear_out(attn_output)

        return attn_output + fsmn_memory


class FunAsrNanoFeedForward(nn.Module):
    """Positionwise feedforward with keys: feed_forward.w_1, feed_forward.w_2."""

    def __init__(self, hidden_size: int, linear_units: int, dropout_rate: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(hidden_size, linear_units)
        self.w_2 = nn.Linear(linear_units, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class FunAsrNanoLayerNorm(nn.LayerNorm):
    """LayerNorm that casts to float32 for numerical stability."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)


class FunAsrNanoEncoderLayer(nn.Module):
    """SANM encoder layer. State dict keys: norm1, norm2, self_attn.*, feed_forward.*"""

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        num_heads: int,
        linear_units: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        kernel_size: int = 11,
        sanm_shift: int = 0,
    ):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size

        self.norm1 = FunAsrNanoLayerNorm(in_size)
        self.norm2 = FunAsrNanoLayerNorm(hidden_size)

        self.self_attn = FunAsrNanoSANMAttention(
            in_features=in_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_dropout_rate=attention_dropout_rate,
            kernel_size=kernel_size,
            sanm_shift=sanm_shift,
        )

        self.feed_forward = FunAsrNanoFeedForward(hidden_size, linear_units, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_out = self.self_attn(x, mask)

        if self.in_size == self.hidden_size:
            x = residual + self.dropout(attn_out)
        else:
            x = self.dropout(attn_out)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))

        return x


class FunAsrNanoEncoder(PreTrainedModel):
    """Fun-ASR-Nano audio encoder (SenseVoiceEncoderSmall / SANM architecture)."""

    config_class = FunAsrNanoEncoderConfig
    main_input_name = "input_features"

    def __init__(self, config: FunAsrNanoEncoderConfig):
        super().__init__(config)

        self.embed = FunAsrNanoSinusoidalPositionEncoder()

        self.encoders0 = nn.ModuleList(
            [
                FunAsrNanoEncoderLayer(
                    in_size=config.input_size,
                    hidden_size=config.output_size,
                    num_heads=config.attention_heads,
                    linear_units=config.linear_units,
                    dropout_rate=config.dropout_rate,
                    attention_dropout_rate=config.attention_dropout_rate,
                    kernel_size=config.kernel_size,
                    sanm_shift=config.sanm_shift,
                )
            ]
        )

        self.encoders = nn.ModuleList(
            [
                FunAsrNanoEncoderLayer(
                    in_size=config.output_size,
                    hidden_size=config.output_size,
                    num_heads=config.attention_heads,
                    linear_units=config.linear_units,
                    dropout_rate=config.dropout_rate,
                    attention_dropout_rate=config.attention_dropout_rate,
                    kernel_size=config.kernel_size,
                    sanm_shift=config.sanm_shift,
                )
                for _ in range(config.num_blocks - 1)
            ]
        )

        self.tp_encoders = nn.ModuleList(
            [
                FunAsrNanoEncoderLayer(
                    in_size=config.output_size,
                    hidden_size=config.output_size,
                    num_heads=config.attention_heads,
                    linear_units=config.linear_units,
                    dropout_rate=config.dropout_rate,
                    attention_dropout_rate=config.attention_dropout_rate,
                    kernel_size=config.kernel_size,
                    sanm_shift=config.sanm_shift,
                )
                for _ in range(config.tp_blocks)
            ]
        )

        self.after_norm = FunAsrNanoLayerNorm(config.output_size)
        self.tp_norm = FunAsrNanoLayerNorm(config.output_size)

        self.post_init()

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lengths: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> BaseModelOutput | tuple:
        hidden_states = input_features.to(dtype=next(self.parameters()).dtype)
        batch_size, max_len, _ = hidden_states.shape

        if feature_lengths is not None:
            mask = torch.arange(max_len, device=hidden_states.device)[None, :] < feature_lengths[:, None]
            mask = mask[:, None, :].to(dtype=hidden_states.dtype)
        else:
            mask = None

        hidden_states = hidden_states * (self.config.output_size**0.5)
        hidden_states = self.embed(hidden_states)

        all_hidden_states = () if output_hidden_states else None

        for layer in self.encoders0:
            hidden_states = layer(hidden_states, mask)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        for layer in self.encoders:
            hidden_states = layer(hidden_states, mask)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        hidden_states = self.after_norm(hidden_states)

        for layer in self.tp_encoders:
            hidden_states = layer(hidden_states, mask)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        hidden_states = self.tp_norm(hidden_states)

        if not return_dict:
            return (hidden_states,) + ((all_hidden_states,) if output_hidden_states else ())

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


# ============================================================================
# Audio Adaptor
# Keys: linear1, linear2, blocks.N.self_attn.linear_{q,k,v,out},
#        blocks.N.feed_forward.w_{1,2}, blocks.N.norm{1,2}
# ============================================================================


class FunAsrNanoAdaptorAttention(nn.Module):
    """Adaptor attention with separate Q/K/V projections matching checkpoint keys."""

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.size()

        q = self.linear_q(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.linear_k(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.linear_v(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)

        if mask is not None:
            # mask shape: (batch, 1, time)
            mask_for_attn = (~mask.bool()).unsqueeze(1)  # (batch, 1, 1, time)
            scores = scores.masked_fill(mask_for_attn, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(b, t, self.hidden_size)
        return self.linear_out(out)


class FunAsrNanoAdaptorLayer(nn.Module):
    """Adaptor transformer layer matching checkpoint structure."""

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0):
        super().__init__()
        self.self_attn = FunAsrNanoAdaptorAttention(hidden_size, num_heads, dropout_rate)
        self.feed_forward = FunAsrNanoFeedForward(hidden_size, hidden_size // 4, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))

        return x


class FunAsrNanoAdaptor(nn.Module):
    """Audio adaptor projecting encoder output to LLM dimension."""

    def __init__(self, config: FunAsrNanoAdaptorConfig):
        super().__init__()
        self.config = config
        self.downsample_rate = config.downsample_rate
        self.use_low_frame_rate = config.use_low_frame_rate

        self.linear1 = nn.Linear(config.encoder_dim * config.downsample_rate, config.ffn_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(config.ffn_dim, config.llm_dim)

        if config.num_layers > 0:
            self.blocks = nn.ModuleList(
                [
                    FunAsrNanoAdaptorLayer(
                        hidden_size=config.llm_dim,
                        num_heads=config.attention_heads,
                        dropout_rate=config.dropout_rate,
                    )
                    for _ in range(config.num_layers)
                ]
            )
        else:
            self.blocks = None

    def forward(self, encoder_out: torch.Tensor, encoder_out_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = encoder_out.size()
        k = self.downsample_rate

        chunk_num = (seq_len - 1) // k + 1
        pad_num = chunk_num * k - seq_len
        if pad_num > 0:
            encoder_out = F.pad(encoder_out, (0, 0, 0, pad_num, 0, 0), value=0.0)

        encoder_out = encoder_out.contiguous().view(batch_size, chunk_num, dim * k)
        x = self.linear1(encoder_out)
        x = self.relu(x)
        x = self.linear2(x)

        output_lens = (encoder_out_lens - 1) // k + 1

        if self.blocks is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device)[None, :] < output_lens[:, None]
            mask = mask[:, None, :].float()  # (batch, 1, time)

            for block in self.blocks:
                x = block(x, mask)

        return x, output_lens


# ============================================================================
# CTC Decoder
# ============================================================================


class FunAsrNanoCtcDecoder(nn.Module):
    """CTC decoder for character-level timestamp prediction."""

    def __init__(self, config: FunAsrNanoCtcConfig):
        super().__init__()
        self.config = config

        self.linear1 = nn.Linear(config.encoder_dim * config.downsample_rate, config.ffn_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(config.ffn_dim, config.decoder_dim)

        if config.num_layers > 0:
            self.blocks = nn.ModuleList(
                [
                    FunAsrNanoAdaptorLayer(
                        hidden_size=config.decoder_dim,
                        num_heads=8,
                        dropout_rate=config.dropout_rate,
                    )
                    for _ in range(config.num_layers)
                ]
            )
        else:
            self.blocks = None

        self.ctc_lo = nn.Linear(config.decoder_dim, config.vocab_size)
        self.blank_id = config.blank_id

    def forward(self, encoder_out: torch.Tensor, encoder_out_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = encoder_out.size()
        k = self.config.downsample_rate

        chunk_num = (seq_len - 1) // k + 1
        pad_num = chunk_num * k - seq_len
        if pad_num > 0:
            encoder_out = F.pad(encoder_out, (0, 0, 0, pad_num, 0, 0), value=0.0)

        x = encoder_out.contiguous().view(batch_size, chunk_num, dim * k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        output_lens = (encoder_out_lens - 1) // k + 1

        if self.blocks is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device)[None, :] < output_lens[:, None]
            mask = mask[:, None, :].float()
            for block in self.blocks:
                x = block(x, mask)

        return x, output_lens

    def log_softmax(self, decoder_out: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.ctc_lo(decoder_out), dim=-1)


# ============================================================================
# Main Model
# ============================================================================


class FunAsrNanoPreTrainedModel(PreTrainedModel):
    """Base class for Fun-ASR-Nano models."""

    config_class = FunAsrNanoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FunAsrNanoEncoderLayer", "FunAsrNanoAdaptorLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        elif isinstance(module, nn.Conv1d):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)


class FunAsrNanoForConditionalGeneration(FunAsrNanoPreTrainedModel, GenerationMixin):
    """
    Fun-ASR-Nano model for speech recognition (conditional generation).

    Architecture: Audio Encoder (SANM) -> Audio Adaptor -> Qwen3 LLM
    Optional: CTC Decoder for timestamp generation.
    """

    def __init__(self, config: FunAsrNanoConfig):
        super().__init__(config)

        self.audio_encoder = FunAsrNanoEncoder(config.audio_encoder_config)
        self.audio_adaptor = FunAsrNanoAdaptor(config.adaptor_config)
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)

        if config.ctc_config is not None:
            self.ctc_decoder = FunAsrNanoCtcDecoder(config.ctc_config)
        else:
            self.ctc_decoder = None

        self.audio_token_index = config.audio_token_index
        self.use_low_frame_rate = getattr(config.adaptor_config, "use_low_frame_rate", True)

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def encode_audio(
        self,
        input_features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode audio through encoder and adaptor.

        Returns: (audio_embeds, audio_embed_lens, encoder_out, encoder_out_lens)
        """
        encoder_outputs = self.audio_encoder(
            input_features=input_features,
            feature_lengths=feature_lengths,
        )
        encoder_out = encoder_outputs.last_hidden_state
        encoder_out_lens = feature_lengths

        audio_embeds, audio_embed_lens = self.audio_adaptor(encoder_out, encoder_out_lens)

        return audio_embeds, audio_embed_lens, encoder_out, encoder_out_lens

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_lengths: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> FunAsrNanoCausalLMOutput | tuple:
        """
        Args:
            input_ids: Token IDs with audio placeholder tokens.
            input_features: Audio features (batch, time, feature_dim) after LFR.
            feature_lengths: Length of each audio feature sequence.
            attention_mask: Token attention mask.
            labels: Labels for language modeling loss (-100 for ignored positions).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            special_audio_mask = input_ids == self.audio_token_index
            if (
                input_features is not None
                and input_ids is not None
                and input_ids.shape[1] != 1
                and special_audio_mask.any()
            ):
                audio_embeds, audio_embed_lens, _, _ = self.encode_audio(input_features, feature_lengths)

                # Mask and scatter audio embeddings into token positions
                special_audio_mask_expanded = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)

                num_audios, max_audio_len, embed_dim = audio_embeds.shape
                audio_len_mask = (
                    torch.arange(max_audio_len, device=audio_embeds.device)[None, :] < audio_embed_lens[:, None]
                )
                flat_audio = audio_embeds[audio_len_mask]
                if special_audio_mask.sum() != flat_audio.shape[0]:
                    raise ValueError(
                        f"Number of audio tokens ({special_audio_mask.sum().item()}) does not match "
                        f"number of audio features ({flat_audio.shape[0]})."
                    )

                inputs_embeds = inputs_embeds.masked_scatter(
                    special_audio_mask_expanded, flat_audio.to(inputs_embeds.dtype)
                )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            **kwargs,
        )

        logits = outputs.logits
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return FunAsrNanoCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        input_features = kwargs.pop("input_features", None)
        feature_lengths = kwargs.pop("feature_lengths", None)
        is_first_iteration = kwargs.get("is_first_iteration", False)

        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        if is_first_iteration or not kwargs.get("use_cache", True):
            model_inputs["input_features"] = input_features
            model_inputs["feature_lengths"] = feature_lengths

        return model_inputs


__all__ = [
    "FunAsrNanoPreTrainedModel",
    "FunAsrNanoEncoder",
    "FunAsrNanoForConditionalGeneration",
]
