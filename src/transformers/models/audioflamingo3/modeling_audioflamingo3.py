"""PyTorch AudioFlamingo3 model."""

from abc import ABC
from collections import deque
from dataclasses import dataclass
from functools import partial
import copy
import math
import os
import os.path as osp
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import file_exists, repo_exists, snapshot_download
from huggingface_hub.utils import HFValidationError

from ... import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    PreTrainedModel,
    PretrainedConfig,
    LlamaForCausalLM,
)
from ...activations import ACT2FN
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...utils import auto_docstring, logging
from .configuration_audioflamingo3 import (
    AudioFlamingo3Config,
    AudioFlamingo3EncoderConfig,
    SoundMultimodalProjectorConfig,
    LlavaConfig,
)

logger = logging.get_logger(__name__)


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
):
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

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}" f" and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        if layer_idx is None and is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
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
    def __init__(self, config: AudioFlamingo3Config):
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
    ) -> torch.Tensor:
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

    def _init_weights(self, module):
        # important: this ported version of AudioFlamingo3 isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed
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


@auto_docstring(
    custom_intro="""
    The audio model from AudioFlamingo3 without any head or projection on top.
    """
)
# Copied from transformers.models.whisper.modeling_whisper.WhisperEncoder with Whisper->AudioFlamingo3
class AudioFlamingo3Encoder(AudioFlamingo3PreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`AudioFlamingo3EncoderLayer`].

    Args:
        config: AudioFlamingo3EncoderConfig
    """

    # Ignore copy
    config: AudioFlamingo3EncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["AudioFlamingo3EncoderLayer"]

    def __init__(self, config: AudioFlamingo3EncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        self.layers = nn.ModuleList([AudioFlamingo3EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        # Ignore copy
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            attention_mask (`torch.Tensor`)`, *optional*):
                AudioFlamingo3 does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"AudioFlamingo3 expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Ignore copy
        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (len(self.layers)), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Ignore copy
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.avg_pooler(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


# -------------------------------------------------------------------------------------------------

__all__ = ["AudioFlamingo3"]


def get_model_config(config):
    default_keys = ["llm_cfg", "sound_tower_cfg", "sound_mm_projector_cfg"]
    if hasattr(config, "_name_or_path") and len(config._name_or_path) >= 2:
        root_path = config._name_or_path
    else:
        root_path = config.resume_path
    # download from huggingface
    if root_path is not None and not osp.exists(root_path):
        try:
            valid_hf_repo = repo_exists(root_path)
        except HFValidationError as e:
            valid_hf_repo = False
        if valid_hf_repo:
            root_path = snapshot_download(root_path)

    return_list = []
    for key in default_keys:
        cfg = getattr(config, key, None)
        if isinstance(cfg, dict):
            try:
                return_list.append(os.path.join(root_path, key[:-4]))
            except:
                raise ValueError(f"Cannot find resume path in config for {key}!")
        elif isinstance(cfg, PretrainedConfig):
            return_list.append(os.path.join(root_path, key[:-4]))
        elif isinstance(cfg, str):
            return_list.append(cfg)

    return return_list


def has_tokenizer(repo_id_or_path: str) -> bool:
    if osp.exists(osp.join(repo_id_or_path, "tokenizer_config.json")):
        return True
    try:
        return repo_exists(repo_id_or_path) and file_exists(repo_id_or_path, "tokenizer_config.json")
    except HFValidationError:
        return False


class LlavaMetaModel(ABC):
    def get_llm(self):
        llm = getattr(self, "llm", None)
        if type(llm) is list:
            llm = llm[0]
        return llm

    def get_sound_tower(self):
        sound_tower = getattr(self, "sound_tower", None)
        if type(sound_tower) is list:
            sound_tower = sound_tower[0]
        return sound_tower

    def get_sound_mm_projector(self):
        sound_mm_projector = getattr(self, "sound_mm_projector", None)
        if type(sound_mm_projector) is list:
            sound_mm_projector = sound_mm_projector[0]
        return sound_mm_projector

    def post_config(self):
        self.training = self.get_llm().training
        ## configuration
        if getattr(self.config, "llm_cfg", None) is None:
            self.config.llm_cfg = self.llm.config
            self.config.speech_tower_cfg = self.speech_tower.config
        if getattr(self.config, "sound_tower_cfg", None) is None:
            self.config.sound_tower_cfg = self.sound_tower.config
        if getattr(self.config, "sound_mm_projector_cfg", None) is None:
            self.config.sound_mm_projector_cfg = self.sound_mm_projector.config

    def get_input_embeddings(self):
        return self.get_llm().get_input_embeddings()

    def get_output_embeddings(self):
        return self.get_llm().get_output_embeddings()

    def resize_token_embeddings(self, embed_size):
        self.get_llm().resize_token_embeddings(embed_size)

    def encode_sound(self, sounds: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        device = self.get_llm().device
        proj_dtype = next(self.get_sound_mm_projector().parameters()).dtype
        sounds = sounds.to(device=device, dtype=proj_dtype)
        if masks is not None:
            masks = masks.to(device)
        sound_features = self.get_sound_tower()(sounds, masks)
        sound_features = self.get_sound_mm_projector()(sound_features.to(dtype=proj_dtype))
        return sound_features


class LlavaMetaForCausalLM(ABC):
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
        media_embeds = deque(self.sound_encoder(media, {}, media_meta["sound_feature_masks"]))
        # This is a workaround to make sure the dummy embeddings are consumed
        # Remove padding
        batch_size = labels.shape[0]

        # Build inverse mapping from token ID to media name
        media_tokens = {}
        for name, token_id in self.tokenizer.media_token_ids.items():
            media_tokens[token_id] = name

        # -------------------------------- #
        num_audio_tokens = torch.stack(media_meta["sound_embed_masks"], dim=0).sum(-1)
        num_audio_tokens = torch.tensor([round(int(x) / 10) * 10 for x in num_audio_tokens])
        num_audios = len(media_embeds)  # length of queue is the number of audios we have in total
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
                # both side is 1, so cannot tell
                left_padding = self.tokenizer.padding_side == "left"
            else:
                # invalid attention_mask
                raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

        # 1. Create a mask to know where special audio tokens are
        special_audio_token_mask = input_ids == self.tokenizer.media_token_ids["sound"]  # hard coded to just work with 'sound'
        num_special_audio_tokens = torch.sum(special_audio_token_mask, dim=-1)

        # In case the Audio model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = text_embeds.device
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)
        num_audio_tokens = num_audio_tokens.to(target_device)
        batch_indices, non_audio_indices = torch.where((input_ids != self.tokenizer.media_token_ids["sound"]) & (attention_mask == 1))

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged audio-text sequence.
        # `special_audio_token_mask` identifies audio tokens. Each audio token will be replaced by `audio_feat_lengths - 1` text tokens.
        # `torch.cumsum` computes how each audio token shifts subsequent text token positions.
        token_placeholder_num = torch.zeros_like(input_ids)
        token_placeholder_num[special_audio_token_mask] = num_audio_tokens.long() - 1
        token_placeholder_num = token_placeholder_num + 1
        new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
        max_token_num = token_placeholder_num.sum(-1).max()
        nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_audio_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
        batch_indices, non_audio_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_audio_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(batch_size, max_token_num, embed_dim, dtype=text_embeds.dtype, device=text_embeds.device)
        final_attention_mask = torch.zeros(batch_size, max_token_num, dtype=attention_mask.dtype, device=text_embeds.device)
        final_input_ids = torch.full((batch_size, max_token_num), self.tokenizer.pad_token_id, dtype=input_ids.dtype, device=text_embeds.device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<audio>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the audio features
        final_embedding[batch_indices, text_to_overwrite] = text_embeds[batch_indices, non_audio_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full_like(final_attention_mask, self.config.ignore_index, dtype=torch.long)  # .to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_audio_indices]

        # 5. Fill the embeddings corresponding to the audios. Anything that is still zeros needs filling
        audio_to_overwrite = torch.full((batch_size, max_token_num), True, dtype=torch.bool, device=text_embeds.device)
        audio_to_overwrite[batch_indices, text_to_overwrite] = False
        seq_indices = torch.arange(max_token_num).unsqueeze(0).to(target_device)
        seq_indices = seq_indices.expand(batch_size, max_token_num)

        if left_padding:
            # exclude padding on the left
            max_token_num = max_token_num.to(target_device)
            val = (max_token_num - seq_indices) <= (token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1))[:, None]
        else:
            # exclude padding on the right
            val = seq_indices < (token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1))[:, None]

        audio_to_overwrite &= val

        if audio_to_overwrite.sum() != num_audio_tokens.sum():
            raise ValueError(
                f"The input provided to the model are wrong. The number of audio tokens is {num_special_audio_tokens} while"
                f" the number of audio given to the model is {num_audios}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[audio_to_overwrite] = masked_audio_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= audio_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
        # # Truncate sequences to `model_max_length` as media embeddings are inserted
        inputs, labels = self.__truncate_sequence(final_embedding, final_labels)
        return self.__batchify_sequence(inputs, labels)

    def __truncate_sequence(self, inputs: List[torch.Tensor], labels: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training and any(len(input) > self.tokenizer.model_max_length for input in inputs):
            warnings.warn(f"Truncating sequences to `model_max_length` ({self.tokenizer.model_max_length}).")
            inputs = [input[: self.tokenizer.model_max_length] for input in inputs]
            labels = [label[: self.tokenizer.model_max_length] for label in labels]
        return inputs, labels

    def __batchify_sequence(self, inputs: List[torch.Tensor], labels: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            if self.tokenizer.padding_side == "right":
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
    ):
        inputs_embeds, _, attention_mask = self._embed(input_ids, media, None, attention_mask, media_meta)
        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generation_kwargs)

    @property
    def default_generation_config(self) -> GenerationConfig:
        generation_config = copy.deepcopy(self.generation_config or GenerationConfig())
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token")
        if generation_config.max_length == GenerationConfig().max_length:
            generation_config.max_length = self.tokenizer.model_max_length
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if generation_config.bos_token_id is None:
            generation_config.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = self.tokenizer.stop_token_ids
        generation_config.max_new_tokens = 512
        return generation_config


class AudioFlamingo3(LlavaMetaModel, LlavaMetaForCausalLM, PreTrainedModel):
    config_class = LlavaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    prepare_inputs_for_generation = LlamaForCausalLM.prepare_inputs_for_generation

    def __init__(self, config: LlavaConfig = None, *args, **kwargs):
        super().__init__(config)
        self.is_loaded = False

        llm_path, sound_tower_cfg, sound_mm_projector_cfg = get_model_config(config)

        llm_cfg = AutoConfig.from_pretrained(llm_path)
        llm = AutoModelForCausalLM.from_pretrained(llm_path, config=llm_cfg, torch_dtype=eval(config.model_dtype), *args, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(llm_path, padding_side="right", use_fast=True, legacy=False)

        # Set stop tokens for the tokenizer
        tokenizer.stop_tokens = ["<|im_end|>"]
        tokenizer.stop_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.stop_tokens)

        # Add media tokens to the tokenizer
        tokenizer.media_tokens = config.media_tokens
        tokenizer.media_token_ids = {}
        for name, token in tokenizer.media_tokens.items():
            tokenizer.add_tokens([token], special_tokens=True)
            tokenizer.media_token_ids[name] = tokenizer.convert_tokens_to_ids(token)

        config.hidden_size = llm.config.hidden_size

        self.llm = llm
        self.tokenizer = tokenizer

        # -----------------------------------------------------------------------------------------

        self.sound_tower = AudioFlamingo3SoundTower(sound_tower_cfg).to(self.llm.device)
        self.sound_mm_projector = SoundMultimodalProjector.from_pretrained(sound_mm_projector_cfg, config, torch_dtype=eval(config.model_dtype)).to(self.llm.device)
        self.sound_encoder = BasicSoundEncoder(parent=self)

        self.post_config()
        self.is_loaded = True

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        devices: Optional[List[int]] = None,
        load_8bit: bool = False,
        load_4bit: bool = False,
        device_map: str = "auto",
        device: str = "cuda",
        *,
        # minimal Hub args
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        local_files_only: bool = False,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ) -> "AudioFlamingo3":
        # — resolve to local path (supports repo_id) ---------------------
        model_path = os.path.expanduser(model_path)
        if not os.path.isdir(model_path):
            # treat as HF repo id
            model_path = snapshot_download(
                repo_id=model_path,
                revision=revision,
                token=token,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
        if os.path.exists(os.path.join(model_path, "model")):
            model_path = os.path.join(model_path, "model")

        # — optional GPU selection --------------------------------------
        if devices is not None:
            assert "max_memory" not in kwargs, "`max_memory` should not be set when `devices` is set"
            kwargs["max_memory"] = {d: torch.cuda.get_device_properties(d).total_memory for d in devices}

        # — device map & quantisation -----------------------------------
        kwargs["device_map"] = {"": device} if device != "cuda" else device_map

        if load_8bit:
            kwargs["load_in_8bit"] = True
        elif load_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = torch.float16

        # — build config -------------------------------------------------
        config = AutoConfig.from_pretrained(model_path)
        config.resume_path = model_path
        config.model_dtype = str(kwargs.pop("torch_dtype", torch.float16))

        # — instantiate & return ----------------------------------------
        model = cls(config=config, low_cpu_mem_usage=True, **kwargs)
        model.eval()
        return model


AutoConfig.register("llava_llama", LlavaConfig)


# -------------------------------------------------------------------------------------------------


class SoundMultimodalProjector(PreTrainedModel):
    config_class = SoundMultimodalProjectorConfig

    def __init__(self, sound_mm_projector_cfg: SoundMultimodalProjectorConfig, config: PretrainedConfig):
        super().__init__(sound_mm_projector_cfg)
        self.layers = nn.Sequential(
            nn.Linear(config.sound_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x, *args, **kwargs):
        return self.layers(x)


AutoConfig.register("sound_mm_projector", SoundMultimodalProjectorConfig)
AutoModel.register(SoundMultimodalProjectorConfig, SoundMultimodalProjector)


# -------------------------------------------------------------------------------------------------


class AudioFlamingo3SoundTower(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()

        self.is_loaded = False
        self.sound_tower_name = model_name_or_path
        self.cfg_only = None
        self.sound_tower = AudioFlamingo3Encoder.from_pretrained(model_name_or_path)
        self.is_loaded = True

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths

    def forward(self, sounds, mask=None):

        if type(sounds) is list:
            sound_features = []
            for sound in sounds:
                # Calculate attention mask
                audio_feat_lengths, audio_output_lengths = self._get_feat_extract_output_lengths(mask.sum(-1))
                # for cases where only one window is there for the audio_clip
                batch_size, _, max_mel_seq_len = sound.shape
                max_seq_len = (max_mel_seq_len - 2) // 2 + 1
                seq_range = torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device).unsqueeze(0).expand(batch_size, max_seq_len)
                lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
                padding_mask = seq_range >= lengths_expand
                audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(batch_size, 1, max_seq_len, max_seq_len)
                audio_attention_mask = audio_attention_mask_.to(dtype=self.sound_tower.conv1.weight.dtype, device=self.sound_tower.conv1.weight.device)
                audio_attention_mask[audio_attention_mask_] = float("-inf")
                # Calculate features
                sound_feature = self.sound_tower(sound, attention_mask=audio_attention_mask)
                sound_feature = sound_feature.to(sound.dtype)
                sound_feature = sound_feature.last_hidden_state
                sound_features.append(sound_feature)
        else:
            # Calculate attention mask
            if len(sounds.shape) == 5:
                sounds = sounds.squeeze(0).squeeze(1)
                mask = mask.squeeze(0)
            audio_feat_lengths, audio_output_lengths = self._get_feat_extract_output_lengths(mask.sum(-1))
            # for cases where only one window is there for the audio_clip
            batch_size, _, max_mel_seq_len = sounds.shape
            max_seq_len = (max_mel_seq_len - 2) // 2 + 1
            seq_range = torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device).unsqueeze(0).expand(batch_size, max_seq_len)
            lengths_expand = audio_feat_lengths.expand(batch_size, max_seq_len)
            padding_mask = seq_range >= lengths_expand
            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(batch_size, 1, max_seq_len, max_seq_len)
            audio_attention_mask = audio_attention_mask_.to(dtype=self.sound_tower.conv1.weight.dtype, device=self.sound_tower.conv1.weight.device)
            audio_attention_mask[audio_attention_mask_] = float("-inf")
            # Calculate features
            sound_features = self.sound_tower(sounds, attention_mask=audio_attention_mask)
            sound_features = sound_features.last_hidden_state
            sound_features = sound_features.to(sounds.dtype)

        return sound_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.sound_tower.dtype

    @property
    def config(self):
        if self.is_loaded:
            return self.sound_tower.config
        else:
            return self.cfg_only

    @property
    def device(self):
        return self.sound_tower.device

    @property
    def hidden_size(self):
        return self.config.hidden_size


# -------------------------------------------------------------------------------------------------


class BasicSoundEncoder(nn.Module):
    def __init__(
        self,
        parent: torch.nn.Module,
        start_tokens: Optional[str] = None,
        end_tokens: Optional[str] = "\n",
    ) -> None:
        super().__init__()
        self._parent = [parent]
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens

    @property
    def parent(self) -> nn.Module:
        return self._parent[0]

    def embed_tokens(self, tokens: Optional[str]) -> Optional[torch.Tensor]:
        if tokens is None:
            return None
        token_ids = self.parent.tokenizer(tokens).input_ids
        token_ids = torch.tensor(token_ids, device=self.parent.device)
        return self.parent.llm.model.embed_tokens(token_ids)

    def _process_features(
        self,
        features: torch.Tensor,
        start_token_embeds: Optional[torch.Tensor],
        end_token_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        features = features.to(self.parent.device)
        if start_token_embeds is not None:
            features = torch.cat([start_token_embeds, features], dim=0)
        if end_token_embeds is not None:
            features = torch.cat([features, end_token_embeds], dim=0)
        return features

    def forward(self, sounds: List[torch.Tensor], config: Dict[str, Any], masks: Dict[str, Any]) -> List[torch.Tensor]:
        sounds = torch.stack(sounds, dim=0)
        masks = torch.stack(masks, dim=0)
        sounds = sounds.to(self.parent.device)
        masks = masks.to(self.parent.device)
        features = self.parent.encode_sound(sounds, masks)
        process_features = partial(
            self._process_features,
            start_token_embeds=self.embed_tokens(self.start_tokens),
            end_token_embeds=self.embed_tokens(self.end_tokens),
        )
        return [process_features(f) for f in features]
