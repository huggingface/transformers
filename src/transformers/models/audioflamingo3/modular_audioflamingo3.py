# coding=utf-8
# Copyright 2025 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
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

"""PyTorch AudioFlamingo3 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ..auto import AutoModelForCausalLM
from ..whisper.modeling_whisper import WhisperAttention, WhisperEncoder, WhisperEncoderLayer
from .configuration_audioflamingo3 import AudioFlamingo3Config, AudioFlamingo3EncoderConfig


logger = logging.get_logger(__name__)


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


@dataclass
class AudioFlamingo3CausalLMOutputWithPast(ModelOutput):
    """
    Output type of :class:`~transformers.AudioFlamingo3ForConditionalGeneration`.

    Args:
        loss (`torch.FloatTensor`, *optional*):
            Next-token prediction loss (returned when `labels` is provided).
        logits (`torch.FloatTensor`, *optional*):
            Scores for each vocabulary token before SoftMax,
            shape `(batch_size, sequence_length, vocab_size)`.
        past_key_values (`Cache`, *optional*):
            Cache to speed up autoregressive decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states of the language model.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Attention maps of the language model.
        attention_mask (`torch.FloatTensor`, *optional*):
            Attention mask passed to the language model.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    attention_mask: Optional[torch.FloatTensor] = None


# --------------------------------------------------------------------------
# Encoder building blocks
# --------------------------------------------------------------------------


class AudioFlamingo3Attention(WhisperAttention):
    """Alias of WhisperAttention kept for configuration/splitting consistency."""

    pass


class AudioFlamingo3EncoderLayer(WhisperEncoderLayer):
    """Alias of WhisperEncoderLayer kept for configuration/splitting consistency."""

    pass


# --------------------------------------------------------------------------
# Base model
# --------------------------------------------------------------------------


class AudioFlamingo3PreTrainedModel(PreTrainedModel):
    """
    Base class with common functionality for AudioFlamingo3 models.
    """

    config_class = AudioFlamingo3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AudioFlamingo3Attention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module) -> None:
        # Initialize modules following config.initializer_range; used for fine-tuning/inference scaffolding.
        std = getattr(self.config, "initializer_range", None)
        if std is None and hasattr(self.config, "audio_config"):
            std = getattr(self.config.audio_config, "initializer_range", 0.02)

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# --------------------------------------------------------------------------
# Audio encoder
# --------------------------------------------------------------------------


class AudioFlamingo3Encoder(WhisperEncoder):
    """
    Audio encoder: Whisper conv front-end, Transformer encoder, average pool (time/2), then LayerNorm.

    Expects `attention_mask` to be `None` or a 4D mask `(B, 1, S, S)` on the *pre-pool* time axis with `-inf` on pads.
    """

    config: AudioFlamingo3EncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["AudioFlamingo3EncoderLayer"]

    def __init__(self, config: AudioFlamingo3EncoderConfig):
        super().__init__(config)
        self.avg_pooler = nn.AvgPool1d(config.avg_pool_kernel_size, stride=config.avg_pool_stride)
        self.post_init()

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, tuple]:
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

        # Conv front-end
        x = nn.functional.gelu(self.conv1(input_features))
        x = nn.functional.gelu(self.conv2(x))  # (B, C, T')

        # Add positions, dropout
        x = x.permute(0, 2, 1)  # (B, S_in, C)
        pos = self.embed_positions.weight
        if pos.shape[0] < x.shape[1]:
            raise ValueError(f"embed_positions shorter than sequence length: {pos.shape[0]} < {x.shape[1]}")
        x = nn.functional.dropout(x + pos[: x.shape[1]], p=self.dropout, training=self.training)

        # Transformer stack
        hs_list = [] if output_hidden_states else None
        attn_list = [] if output_attentions else None
        h = x
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                hs_list.append(h)
            to_drop = self.training and (torch.rand([]) < self.layerdrop)
            if to_drop:
                out = (h, None)
            else:
                out = layer(
                    h,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )
                h = out[0]
            if output_attentions:
                attn_list.append(out[1])

        # AvgPool (time/2) + LayerNorm
        prepool = h
        h = h.permute(0, 2, 1)
        h = self.avg_pooler(h).permute(0, 2, 1)  # (B, S_out, C)
        h = self.layer_norm(h)

        if output_hidden_states:
            hs_list.append(prepool)
            hs_list.append(h)

        if not return_dict:
            outs = (
                h,
                tuple(hs_list) if hs_list is not None else None,
                tuple(attn_list) if attn_list is not None else None,
            )
            return tuple(v for v in outs if v is not None)

        return BaseModelOutput(
            last_hidden_state=h,
            hidden_states=tuple(hs_list) if hs_list is not None else None,
            attentions=tuple(attn_list) if attn_list is not None else None,
        )

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Compute (pre-conv) and (post-pool) sequence lengths given mel frame lengths.
        Matches the conv/pool schedule used in `forward`.
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


# --------------------------------------------------------------------------
# Projector
# --------------------------------------------------------------------------


class AudioFlamingo3MultiModalProjector(nn.Module):
    """
    Audio adaptor (a small MLP) that projects AudioFlamingo3Encoder (AF-Whisper)
    features to the LLM embedding space so they can replace `<sound>` tokens.
    """

    def __init__(self, config: AudioFlamingo3Config) -> None:
        super().__init__()
        d_audio = config.audio_config.d_model
        d_text = config.text_config.hidden_size
        self.layers = nn.ModuleList([nn.Linear(d_audio, d_text), nn.GELU(), nn.Linear(d_text, d_text)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# --------------------------------------------------------------------------
# Conditional generation model
# --------------------------------------------------------------------------


class AudioFlamingo3ForConditionalGeneration(AudioFlamingo3PreTrainedModel, GenerationMixin):
    """
    AudioFlamingo3 model composed of an audio encoder, a projection to the LM hidden size, and a causal LM.

    The audio-text fusion is performed by *replacing* occurrences of the `<sound>` token with per-frame audio embeddings,
    without changing sequence length. The number of `<sound>` tokens is expected to match the *post-pool* frame count
    computed by the processor.
    """

    config_class = AudioFlamingo3Config

    def __init__(self, config: AudioFlamingo3Config):
        super().__init__(config)
        # Language model
        self.llm = AutoModelForCausalLM.from_config(config.text_config)
        # Audio encoder (explicitly instantiate our class to guarantee helper availability)
        self.sound_tower = AudioFlamingo3Encoder(config.audio_config)
        # Projection to LM hidden size
        self.sound_mm_projector = AudioFlamingo3MultiModalProjector(config)

        # Common IDs / limits
        self.padding_side = config.padding_side
        self.pad_token_id = config.pad_token_id
        self.model_max_length = config.model_max_length
        self.eos_token_id = config.eos_token_id
        self.bos_token_id = config.bos_token_id
        self.sound_token_id = config.sound_token_id

        self.post_init()

    # --- Embedding plumbing (forward to LM) ---
    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.llm.set_output_embeddings(value)

    def set_decoder(self, decoder):
        self.llm.set_decoder(decoder)

    def get_decoder(self):
        return self.llm.get_decoder()

    # --- Forward ---
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,  # (#windows, n_mels, T_mel)
        attention_mask: Optional[torch.Tensor] = None,  # (B, L)
        feature_attention_mask: Optional[torch.Tensor] = None,  # (#windows, T_mel)
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, AudioFlamingo3CausalLMOutputWithPast]:
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        # Text embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Replace <sound> token slots with audio features (no length change)
        if input_features is not None and input_ids is not None and input_ids.shape[1] != 1:
            dev = next(self.sound_tower.parameters()).device

            input_features = input_features.to(dev)
            if feature_attention_mask is None:
                raise ValueError("`feature_attention_mask` is required when `input_features` is provided.")
            feature_attention_mask = feature_attention_mask.to(dev)

            # Compute pre/post lengths (mel -> conv -> pool)
            Lmel = feature_attention_mask.sum(-1)  # (#windows,)
            pre_lengths, post_lengths = self.sound_tower._get_feat_extract_output_lengths(Lmel)
            pre_lengths = pre_lengths.to(dtype=torch.long)
            post_lengths = post_lengths.to(dtype=torch.long)

            # Build 4D encoder mask on pre-pool axis with -inf on pads
            _, _, T_mel_max = input_features.shape
            S_in_max = (T_mel_max - 1) // 2 + 1
            seq = (
                torch.arange(S_in_max, dtype=torch.long, device=pre_lengths.device)
                .unsqueeze(0)
                .expand(pre_lengths.shape[0], S_in_max)
            )
            pad_bool = seq >= pre_lengths.unsqueeze(1)  # (N, S_in_max)
            enc_mask_bool = pad_bool.view(pre_lengths.shape[0], 1, 1, S_in_max).expand(
                pre_lengths.shape[0], 1, S_in_max, S_in_max
            )
            enc_mask = enc_mask_bool.to(dtype=self.sound_tower.conv1.weight.dtype, device=dev)
            enc_mask[enc_mask_bool] = float("-inf")

            # Encode audio -> project -> flatten valid frames
            enc_out = self.sound_tower(input_features, attention_mask=enc_mask)
            post = enc_out.last_hidden_state  # (#windows, S_out, C)
            audio_feats = self.sound_mm_projector(post)  # (#windows, S_out, D)

            N, S_out_max, D = audio_feats.shape
            valid_mask = torch.arange(S_out_max, device=post_lengths.device)[None, :] < post_lengths[:, None]
            flat_audio = audio_feats[valid_mask]  # (sum(post_lengths), D)

            # Robust per-sample assignment (no masked_scatter interleaving)
            with torch.no_grad():
                per_sample_counts = (input_ids == self.sound_token_id).sum(dim=1).tolist()
                total_tokens = int(sum(per_sample_counts))
                total_frames = int(flat_audio.shape[0])
                if total_tokens != total_frames:
                    raise ValueError(
                        f"Audio tokens and features mismatch: tokens={total_tokens}, frames={total_frames}. "
                        "Ensure the processor expands <sound> by the post-pool frame count."
                    )

                chunks = []
                off = 0
                for sz in per_sample_counts:
                    chunks.append(flat_audio[off : off + sz])
                    off += sz
                assert off == total_frames

            bsz = input_ids.size(0)
            for b in range(bsz):
                pos_b = (input_ids[b] == self.sound_token_id).nonzero(as_tuple=False).squeeze(-1)  # (T_b,)
                if pos_b.numel() == 0:
                    continue
                feats_b = chunks[b].to(inputs_embeds.device, dtype=inputs_embeds.dtype)  # (T_b, D)
                if feats_b.size(0) != pos_b.numel():
                    raise RuntimeError(f"Sample {b}: token/feature mismatch {pos_b.numel()} vs {feats_b.size(0)}")
                inputs_embeds[b, pos_b, :] = feats_b

        # Language model forward
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        logits = outputs[0]

        # Optional loss
        loss = None
        if labels is not None:
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return AudioFlamingo3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask,
        )

    # --- Generation helpers ---
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
        Pass `input_features`/`feature_attention_mask` only on the first step of generation.
        """
        input_features = kwargs.pop("input_features", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        is_first = model_inputs.get("past_key_values", None) is None or (
            isinstance(model_inputs.get("cache_position", None), torch.Tensor)
            and model_inputs["cache_position"].numel() > 0
            and int(model_inputs["cache_position"][0].item()) == 0
        )
        if is_first:
            if input_features is not None:
                model_inputs["input_features"] = input_features
            if feature_attention_mask is not None:
                model_inputs["feature_attention_mask"] = feature_attention_mask
        return model_inputs


__all__ = ["AudioFlamingo3ForConditionalGeneration", "AudioFlamingo3PreTrainedModel", "AudioFlamingo3Encoder"]
