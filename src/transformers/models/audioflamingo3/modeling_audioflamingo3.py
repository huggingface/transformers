"""PyTorch AudioFlamingo3 model."""

from collections import deque
import copy
import math
import os.path as osp
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from huggingface_hub.utils import HFValidationError

from ... import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    GenerationMixin,
    AutoModel,
)
from ...activations import ACT2FN
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import ModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...utils import auto_docstring, logging
from .configuration_audioflamingo3 import (
    AudioFlamingo3EncoderConfig,
    AudioFlamingo3Config,
)

logger = logging.get_logger(__name__)


@dataclass
class AudioFlamingo3CausalLMOutputWithPast(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


# Copied from transformers.models.whisper.modeling_whisper.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None and attention_mask.ndim == 4:
        attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask.view(1, -1, 1, 1)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class AudioFlamingo3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.whisper.modeling_whisper.WhisperAttention.__init__ with Whisper->AudioFlamingo3
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        layer_idx: Optional[int] = None,
        config: Optional[AudioFlamingo3Config] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.size()

        # Scaling is susceptible to floating point arithmetics' inprecisions
        # which can lead to different results (this is dependent from model
        # to model, e.g. whisper is one such case). We therefore keep the
        # original order of scaling to follow the original implementation
        # and enforce no scaling (1.0) in the attention call below.
        query_states = self._shape(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,
            output_attentions=output_attentions,
            head_mask=layer_head_mask,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.whisper.modeling_whisper.WhisperEncoderLayer with Whisper->AudioFlamingo3, WHISPER->AUDIOFLAMINGO3
class AudioFlamingo3EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: AudioFlamingo3Config) -> None:
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = AudioFlamingo3Attention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states, attn_weights


@auto_docstring
class AudioFlamingo3PreTrainedModel(PreTrainedModel):
    config: AudioFlamingo3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AudioFlamingo3Attention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range if hasattr(self.config, "initializer_range") else self.config.audio_config.initializer_range

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


class AudioFlamingo3Encoder(AudioFlamingo3PreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers.
    This version folds the old "AudioFlamingo3SoundTower" wrapper functionality directly into the encoder.

    Use cases:
      - HF-native: forward(input_features=..., attention_mask=...) -> BaseModelOutput
      - Tower-style (old wrapper): forward_tower(sounds, mask) -> last_hidden_state
    """

    # keep HF typing and split rules
    config: AudioFlamingo3EncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["AudioFlamingo3EncoderLayer"]

    def __init__(self, config: AudioFlamingo3EncoderConfig) -> None:
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # frontend
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        # fixed positional embeddings (non-trainable, like Whisper)
        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        # transformer
        self.layers = nn.ModuleList([AudioFlamingo3EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        # additional pooling to match your prior wrapper’s behavior
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

        self.gradient_checkpointing = False
        self.post_init()

    # ----------------------------
    # Compatibility helpers (tower)
    # ----------------------------

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Computes output lengths after the two convs:
          conv1: stride=1
          conv2: stride=2
        And returns both the feature length after conv1 and after conv2 (encoder S).
        """
        input_lengths = (input_lengths - 1) // 2 + 1  # conv2 path as in your previous code (kept verbatim)
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths

    def _build_square_attn_mask(self, mask_1d: torch.Tensor, max_mel_seq_len: int) -> torch.Tensor:
        """
        Build (B,1,S,S) attention mask with -inf on padded positions for Whisper-style encoders.

        mask_1d: (B, T_mel) boolean/0-1 mask indicating valid mel frames.
        max_mel_seq_len: T_mel
        """
        audio_feat_lengths, _ = self._get_feat_extract_output_lengths(mask_1d.sum(-1))
        B = mask_1d.shape[0]
        S = (max_mel_seq_len - 2) // 2 + 1

        seq_range = torch.arange(S, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device).unsqueeze(0).expand(B, S)
        lengths_expand = audio_feat_lengths.expand(B, S)
        padding_mask = seq_range >= lengths_expand  # (B, S) True => pad

        square = padding_mask.view(B, 1, 1, S).expand(B, 1, S, S)
        attn = square.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)
        attn[square] = float("-inf")
        return attn

    @torch.no_grad()
    def forward_tower(self, sounds: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Wrapper-compatible entry point:
        sounds: (B, num_mel_bins, T_mel) or (1,1,B,num_mel_bins,T_mel)
        mask:   (B, T_mel) 0/1 mask for valid mel frames (required)

        Returns:
          last_hidden_state: (B, S, d_model) in sounds.dtype
        """
        if sounds.ndim == 5:
            sounds = sounds.squeeze(0).squeeze(1)  # -> (B, M, T)
            if mask is not None:
                mask = mask.squeeze(0)

        if mask is None:
            raise ValueError("forward_tower requires a frame mask of shape (B, T_mel).")

        B, M, T_mel = sounds.shape
        if M != self.num_mel_bins:
            raise ValueError(f"Expected sounds with num_mel_bins={self.num_mel_bins}, got {M}.")

        attn_mask = self._build_square_attn_mask(mask, max_mel_seq_len=T_mel)
        out = self.forward(input_features=sounds, attention_mask=attn_mask)
        return out.last_hidden_state.to(sounds.dtype)

    # HF-required embeddings API (kept)
    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.conv1 = value

    # ----------------------------
    # Core HF forward
    # ----------------------------
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[AudioFlamingo3CausalLMOutputWithPast, Tuple[torch.Tensor, Optional[tuple[torch.Tensor]], Optional[tuple[torch.Tensor]]]]:
        """
        HF-native forward with mel-spectrogram features.
        input_features: (B, num_mel_bins, T_mel)
        attention_mask: (B, 1, S, S) with -inf on padded positions (optional)
        """
        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"AudioFlamingo3 expects the mel input features to be of length {expected_seq_length}, " f"but found {input_features.shape[-1]}. Pad/truncate mel features to {expected_seq_length}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # dtype/device normalization (match conv weights)
        x = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)  # (B, M, T)

        # frontend convs
        x = nn.functional.gelu(self.conv1(x))
        x = nn.functional.gelu(self.conv2(x))  # (B, d_model, T/2)

        # time-major for transformer: (B, T', C)
        x = x.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight  # (max_source_positions, C)
        # broadcast add positional embeddings (trim if needed)
        if embed_pos.shape[0] < x.shape[1]:
            raise ValueError(f"embed_positions shorter than sequence length: {embed_pos.shape[0]} < {x.shape[1]}")
        x = x + embed_pos[: x.shape[1]]
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        if head_mask is not None:
            assert head_mask.size(0) == len(self.layers), f"head_mask should have {len(self.layers)} layers, but has {head_mask.size(0)}."

        hidden_states = x
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            to_drop = False
            if self.training and torch.rand([]) < self.layerdrop:
                to_drop = True

            if to_drop:
                layer_outputs = (hidden_states, None)
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attns = all_attns + (layer_outputs[1],)

        # match prior wrapper’s pooling path
        hs = hidden_states.permute(0, 2, 1)  # (B, C, S)
        hs = self.avg_pooler(hs)  # downsample in time
        hs = hs.permute(0, 2, 1)  # (B, S', C)
        hs = self.layer_norm(hs)

        if output_hidden_states:
            encoder_states = encoder_states + (hs,)

        if not return_dict:
            return tuple(v for v in [hs, encoder_states, all_attns] if v is not None)

        return AudioFlamingo3CausalLMOutputWithPast(last_hidden_state=hs, hidden_states=encoder_states, attentions=all_attns)

    # ----------------------------
    # Legacy convenience properties
    # ----------------------------
    @property
    def device(self) -> torch.device:
        return self.conv1.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    @property
    def hidden_size(self) -> int:
        return self.config.d_model


class AudioFlamingo3ForConditionalGeneration(AudioFlamingo3PreTrainedModel, GenerationMixin):
    config_class = AudioFlamingo3Config

    def __init__(self, config: Optional[AudioFlamingo3Config] = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(config)

        self.llm = AutoModelForCausalLM.from_config(config.llm_cfg)
        self.sound_tower = AutoModel.from_config(config.sound_tower_cfg)
        self.sound_mm_projector = AudioFlamingo3MultiModalProjector(config)

        self.media_tokens = config.media_tokens
        self.padding_side = config.padding_side
        self.pad_token_id = config.pad_token_id
        self.model_max_length = config.model_max_length
        self.eos_token_id = config.eos_token_id
        self.bos_token_id = config.bos_token_id
        self.sound_token_id = config.sound_token_id
        self.end_newline_token_id = config.end_newline_token_id

    def _process_features(
        self,
        features: torch.Tensor,
        start_token_embeds: Optional[torch.Tensor],
        end_token_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        out = features.to(self.llm.device)
        if start_token_embeds is not None:
            out = torch.cat([start_token_embeds, out], dim=0)
        if end_token_embeds is not None:
            out = torch.cat([out, end_token_embeds], dim=0)
        return out

    def _sound_features(
        self,
        sounds: List[torch.Tensor],
        masks: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        sounds = torch.stack(sounds, dim=0).to(self.llm.device)
        masks = torch.stack(masks, dim=0).to(self.llm.device)
        feats = self.encode_sound(sounds, masks)  # (B, S, D)

        end_emb = self.llm.model.embed_tokens(torch.tensor([self.end_newline_token_id], device=self.llm.device))
        return [self._process_features(f, None, end_emb) for f in feats]

    def encode_sound(self, sounds: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = self.llm.device
        proj_dtype = next(self.sound_mm_projector.parameters()).dtype
        sounds = sounds.to(device=device, dtype=proj_dtype)
        masks = masks.to(device) if masks is not None else None

        feats = self.sound_tower.forward_tower(sounds, masks).to(dtype=proj_dtype)
        return self.sound_mm_projector(feats)

    def _embed(
        self,
        input_ids: torch.Tensor,
        media: List[torch.Tensor],
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        media_meta: Dict[str, Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = labels if labels is not None else torch.full_like(input_ids, self.config.ignore_index)
        attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)

        # Extract text and media embeddings
        text_embeds = self.llm.model.embed_tokens(input_ids)

        media_embeds = deque(self._sound_features(media, media_meta["sound_feature_masks"]))

        batch_size = labels.shape[0]

        num_audio_tokens = torch.stack(media_meta["sound_embed_masks"], dim=0).sum(-1)
        num_audio_tokens = torch.tensor([round(int(x) / 10) * 10 for x in num_audio_tokens])
        num_audios = len(media_embeds)  # number of total audios
        max_audio_tokens, embed_dim = media_embeds[0].shape

        audio_features_mask = torch.arange(max_audio_tokens).expand(num_audios, max_audio_tokens).to(num_audio_tokens.device) < num_audio_tokens.unsqueeze(1)

        audio_embeds = []
        while media_embeds:
            audio_embeds.append(media_embeds.popleft())
        audio_embeds = torch.stack(audio_embeds, dim=0)

        masked_audio_features = audio_embeds[audio_features_mask].view(-1, embed_dim)
        batch_size, sequence_length = input_ids.shape
        _left_padding = torch.any(attention_mask[:, 0] == 0)
        _right_padding = torch.any(attention_mask[:, -1] == 0)

        left_padding = True
        if batch_size > 1:
            if _left_padding and not _right_padding:
                left_padding = True
            elif not _left_padding and _right_padding:
                left_padding = False
            elif not _left_padding and not _right_padding:
                left_padding = self.padding_side == "left"
            else:
                raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

        # 1. Mask of special audio tokens (uses media_tokens["sound"])
        special_audio_token_mask = input_ids == self.sound_token_id
        num_special_audio_tokens = torch.sum(special_audio_token_mask, dim=-1)

        # devices
        target_device = text_embeds.device
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)
        num_audio_tokens = num_audio_tokens.to(target_device)
        batch_indices, non_audio_indices = torch.where((input_ids != self.sound_token_id) & (attention_mask == 1))

        # 2. positions where text should be written
        token_placeholder_num = torch.zeros_like(input_ids)
        token_placeholder_num[special_audio_token_mask] = num_audio_tokens.long() - 1
        token_placeholder_num = token_placeholder_num + 1
        new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
        max_token_num = token_placeholder_num.sum(-1).max()
        nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_audio_pad[:, None]
        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
        batch_indices, non_audio_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_audio_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 3. final padded embedding containers
        final_embedding = torch.zeros(batch_size, max_token_num, embed_dim, dtype=text_embeds.dtype, device=text_embeds.device)
        final_attention_mask = torch.zeros(batch_size, max_token_num, dtype=attention_mask.dtype, device=text_embeds.device)
        final_input_ids = torch.full((batch_size, max_token_num), self.pad_token_id, dtype=input_ids.dtype, device=text_embeds.device)

        # 4. scatter text
        final_embedding[batch_indices, text_to_overwrite] = text_embeds[batch_indices, non_audio_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full_like(final_attention_mask, self.config.ignore_index, dtype=torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_audio_indices]

        # 5. scatter audio features
        audio_to_overwrite = torch.full((batch_size, max_token_num), True, dtype=torch.bool, device=text_embeds.device)
        audio_to_overwrite[batch_indices, text_to_overwrite] = False
        seq_indices = torch.arange(max_token_num).unsqueeze(0).to(target_device)
        seq_indices = seq_indices.expand(batch_size, max_token_num)

        if left_padding:
            max_token_num = max_token_num.to(target_device)
            val = (max_token_num - seq_indices) <= (token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1))[:, None]
        else:
            val = seq_indices < (token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1))[:, None]

        audio_to_overwrite &= val

        if audio_to_overwrite.sum() != num_audio_tokens.sum():
            raise ValueError(f"Bad inputs: #audio tokens={num_special_audio_tokens} vs #audios={num_audios}. " "Indexing would break.")

        final_embedding[audio_to_overwrite] = masked_audio_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= audio_to_overwrite

        # Truncate & batchify
        inputs, labels = self.__truncate_sequence(final_embedding, final_labels)
        return self.__batchify_sequence(inputs, labels)

    def __truncate_sequence(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
        labels: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self.training and any(len(input) > self.model_max_length for input in inputs):
            warnings.warn(f"Truncating sequences to `model_max_length` ({self.model_max_length}).")
            inputs = [input[: self.model_max_length] for input in inputs]
            labels = [label[: self.model_max_length] for label in labels]
        return list(inputs), list(labels)

    def __batchify_sequence(
        self,
        inputs: List[torch.Tensor],
        labels: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(inputs)
        device = inputs[0].device
        hidden_size = inputs[0].shape[1]
        max_length = max(inputs[k].shape[0] for k in range(batch_size))
        attention_mask = torch.ones((batch_size, max_length), dtype=torch.bool, device=device)

        inputs_p, labels_p = [], []
        for k in range(batch_size):
            size_pk = max_length - inputs[k].shape[0]
            inputs_pk = torch.zeros((size_pk, hidden_size), dtype=inputs[k].dtype, device=device)
            labels_pk = torch.full((size_pk,), self.config.ignore_index, dtype=labels[k].dtype, device=device)
            if self.padding_side == "right":
                attention_mask[k, inputs[k].shape[0] :] = False
                inputs_pk = torch.cat([inputs[k], inputs_pk], dim=0)
                labels_pk = torch.cat([labels[k], labels_pk], dim=0)
            else:
                attention_mask[k, : -inputs[k].shape[0]] = False
                inputs_pk = torch.cat([inputs_pk, inputs[k]], dim=0)
                labels_pk = torch.cat([labels_pk, labels[k]], dim=0)
            inputs_p.append(inputs_pk)
            labels_p.append(labels_pk)

        inputs = torch.stack(inputs_p, dim=0)
        labels = torch.stack(labels_p, dim=0)
        return inputs, labels, attention_mask

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        media: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        media_meta: Dict[str, Dict[str, Any]] = None,
        **generation_kwargs,
    ) -> torch.LongTensor:
        inputs_embeds, _, attention_mask = self._embed(input_ids, media, None, attention_mask, media_meta)
        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generation_kwargs)

    @property
    def default_generation_config(self) -> GenerationConfig:
        generation_config = copy.deepcopy(self.generation_config or GenerationConfig())
        if self.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token")
        if generation_config.max_length == GenerationConfig().max_length:
            generation_config.max_length = self.model_max_length
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = self.pad_token_id or self.eos_token_id
        if generation_config.bos_token_id is None:
            generation_config.bos_token_id = self.bos_token_id or self.eos_token_id
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = [self.eos_token_id]
        generation_config.do_sample = False
        generation_config.max_new_tokens = 2048
        return generation_config


class AudioFlamingo3MultiModalProjector(nn.Module):
    def __init__(self, config: AudioFlamingo3Config) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.sound_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.layers(x)


__all__ = ["AudioFlamingo3ForConditionalGeneration", "AudioFlamingo3PreTrainedModel", "AudioFlamingo3Encoder"]
