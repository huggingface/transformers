# coding=utf-8
# Copyright 2025 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...generation import GenerationMixin
from ...modeling_outputs import Seq2SeqLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    auto_docstring,
    logging,
    replace_return_docstrings,
)
from ..bart.modeling_bart import (
    BartDecoder,
    BartDecoderLayer,
    BartEncoder,
    BartEncoderLayer,
    BartForConditionalGeneration,
    BartModel,
    BartPreTrainedModel,
    BartScaledWordEmbedding,
)
from ..beit.modeling_beit import BeitDropPath
from ..detr.modeling_detr import DetrLearnedPositionEmbedding
from .configuration_florence2 import Florence2Config, Florence2LanguageConfig, Florence2VisionConfig


_CHECKPOINT_FOR_DOC = "microsoft/Florence-2-large"

logger = logging.get_logger(__name__)


class Florence2VisionDropPath(BeitDropPath):
    pass


class Florence2VisionLearnedAbsolutePositionEmbedding2D(DetrLearnedPositionEmbedding):
    def __init__(self, embedding_dim: int = 256, num_pos: int = 50):
        super().__init__()
        embedding_dim = embedding_dim
        self.row_embeddings = nn.Embedding(num_pos, embedding_dim // 2)
        self.column_embeddings = nn.Embedding(num_pos, embedding_dim - (embedding_dim // 2))


class Florence2VisionPositionalEmbeddingCosine1D(nn.Module):
    """
    This class implements a very simple positional encoding. It follows closely
    the encoder from the link below:
    https://pytorch.org/tutorials/beginner/translation_transformer.html

    Args:
        embed_dim: The dimension of the embeddings.
        dropout_prob: The dropout probability.
        max_seq_len: The maximum length to precompute the positional encodings.
    """

    def __init__(self, embed_dim: int = 512, max_seq_len: int = 1024) -> None:
        super(Florence2VisionPositionalEmbeddingCosine1D, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        pos_idx_to_embed = torch.empty((self.max_seq_len, self.embed_dim))
        sine, cosine = Florence2VisionPositionalEmbeddingCosine1D.get_sinusoid_embeddings()
        pos_idx_to_embed[:, 0::2] = sine
        pos_idx_to_embed[:, 1::2] = cosine
        # Save the positional embeddings in a constant buffer.
        self.register_buffer("pos_idx_to_embed", pos_idx_to_embed)

    @staticmethod
    def get_sinusoid_embeddings(max_positions: int, embed_dim: int):
        half_dim = embed_dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(max_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        return torch.sin(emb), torch.cos(emb)

    def forward(self, seq_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_embeds: The sequence embeddings in order. Allowed size:
                1. [T, D], where T is the length of the sequence, and D is the
                frame embedding dimension.
                2. [B, T, D], where B is the batch size and T and D are the
                same as above.

        Returns a tensor of with the same dimensions as the input: i.e.,
        [1, T, D] or [T, D].
        """
        len_seq = seq_embeds.size(-2)
        if len_seq > self.max_seq_len:
            raise ValueError(f"Maximum sequence length {self.max_seq_len}, got {len_seq}")
        pos_embeds = self.pos_idx_to_embed[0 : seq_embeds.size(-2), :]
        return pos_embeds


class Florence2VisionMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Florence2VisionDepthWiseConv2d(nn.Module):
    def __init__(
        self,
        dim_in: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            dim_in,
            dim_in,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim_in,
        )

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states


class Florence2VisionConvEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        config: Florence2VisionConfig,
        stage_idx: int,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size[stage_idx]
        self.in_chans = config.in_chans if stage_idx == 0 else config.dim_embed[stage_idx - 1]
        self.dim_embed = config.dim_embed[stage_idx]
        self.stride = config.patch_stride[stage_idx]
        self.padding = config.patch_padding[stage_idx]
        self.pre_norm = config.patch_prenorm[stage_idx]

        self.conv = nn.Conv2d(
            self.in_chans,
            self.dim_embed,
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=self.padding,
        )

        dim_norm = self.in_chans if self.pre_norm else self.dim_embed
        self.norm = norm_layer(dim_norm) if norm_layer else None

    def forward(self, hidden_states: torch.Tensor):
        B, C, H, W = hidden_states.shape

        hidden_states = self.conv(hidden_states)

        _, _, H, W = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            hidden_states.shape[0], H * W, hidden_states.shape[1]
        )
        if self.norm and not self.pre_norm:
            hidden_states = self.norm(hidden_states)

        return hidden_states


class Florence2VisionChannelAttention(nn.Module):
    def __init__(self, dim: int, groups: int = 8, qkv_bias: bool = True):
        super().__init__()

        self.groups = groups
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, hidden_states: torch.Tensor):
        B, N, C = hidden_states.shape

        qkv = self.qkv(hidden_states).reshape(B, N, 3, self.groups, C // self.groups).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        # Dynamic scale
        query = query * N**-0.5
        attn_weights = query.transpose(-1, -2) @ key
        attn_weights = attn_weights.softmax(dim=-1)
        hidden_states = (attn_weights @ value.transpose(-1, -2)).transpose(-1, -2)
        hidden_states = hidden_states.transpose(1, 2).reshape(B, N, C)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class Florence2VisionChannelBlock(nn.Module):
    def __init__(
        self,
        config: Florence2VisionConfig,
        stage_idx: int,
        channel_drop_path_rate: float,
        kernel_size: int = 3,
        act_layer: nn.Module = nn.GELU,
    ):
        super().__init__()

        self.config = config
        dim_in = config.dim_embed[stage_idx]

        self.conv1 = nn.Conv2d(
            dim_in,
            dim_in,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim_in,
        )
        self.norm1 = nn.LayerNorm(config.dim_embed[stage_idx])
        self.channel_attn = Florence2VisionChannelAttention(
            config.dim_embed[stage_idx],
            groups=config.num_groups[stage_idx],
            qkv_bias=config.qkv_bias,
        )
        self.drop_path1 = (
            Florence2VisionDropPath(channel_drop_path_rate) if channel_drop_path_rate > 0.0 else nn.Identity()
        )

        self.conv1 = nn.Conv2d(
            dim_in,
            dim_in,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim_in,
        )
        self.norm2 = nn.LayerNorm(config.dim_embed[stage_idx])
        self.ffn = Florence2VisionMLP(
            in_features=config.dim_embed[stage_idx],
            hidden_features=int(config.dim_embed[stage_idx] * config.mlp_ratio),
            act_layer=act_layer,
        )
        self.drop_path2 = (
            Florence2VisionDropPath(channel_drop_path_rate) if channel_drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, hidden_states: torch.Tensor):
        B, C, H, W = hidden_states.shape

        hidden_states = self.conv1(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        cur = self.norm1(hidden_states)
        cur = self.channel_attn(cur)
        hidden_states = hidden_states + self.drop_path(cur)
        hidden_states = hidden_states.transpose(1, 2).view(B, C, H, W)

        hidden_states = self.conv2(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        cur = self.norm1(hidden_states)
        cur = self.ffn(cur)
        hidden_states = hidden_states + self.drop_path(cur)
        hidden_states = hidden_states.transpose(1, 2).view(B, C, H, W)

        return hidden_states


def window_partition(hidden_states: torch.Tensor, window_size: int):
    B, H, W, C = hidden_states.shape
    hidden_states = hidden_states.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    C = windows.shape[-1]
    hidden_states = windows.view(-1, H // window_size, W // window_size, window_size, window_size, C)
    hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return hidden_states


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Mask heads if we want to
    if attention_mask is not None:
        attn_weights = attn_weights * attention_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Florence2VisionWindowAttention(nn.Module):
    def __init__(
        self,
        config: Florence2VisionConfig,
        dim: int,
        num_heads: int,
        window_size: int,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_states: torch.Tensor):
        B, C, H, W = hidden_states.shape

        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        hidden_states = F.pad(hidden_states, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = hidden_states.shape

        window_contexts = window_partition(hidden_states, self.window_size)
        window_contexts = window_contexts.view(-1, self.window_size * self.window_size, C)

        B_, N, C = window_contexts.shape
        qkv = self.qkv(window_contexts).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        window_contexts = eager_attention_forward(query, key, value, scale=self.scale)
        window_contexts = window_contexts.view(B_, N, C)
        window_contexts = self.proj(window_contexts)

        # merge windows
        window_contexts = window_contexts.view(-1, self.window_size, self.window_size, C)
        hidden_states = window_reverse(window_contexts, self.window_size, Hp, Wp)

        hidden_states = hidden_states[:, :H, :W, :].contiguous()
        hidden_states = hidden_states.view(B, H * W, C)

        return hidden_states


class Florence2VisionSpatialBlock(nn.Module):
    def __init__(
        self,
        config: Florence2VisionConfig,
        stage_idx: int,
        drop_path_rate: float,
        act_layer: nn.Module = nn.GELU,
    ):
        super().__init__()

        self.conv1 = Florence2VisionDepthWiseConv2d(config.dim_embed[stage_idx])
        self.norm1 = nn.LayerNorm(config.dim_embed[stage_idx])
        self.window_attn = (
            Florence2VisionWindowAttention(
                config,
                config.dim_embed[stage_idx],
                config.num_heads[stage_idx],
                config.window_size,
                config.qkv_bias,
            ),
        )
        self.drop_path1 = Florence2VisionDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.conv2 = Florence2VisionDepthWiseConv2d(config.dim_embed[stage_idx])
        self.norm2 = nn.LayerNorm(config.dim_embed[stage_idx])
        self.ffn = Florence2VisionMLP(
            in_features=config.dim_embed[stage_idx],
            hidden_features=int(config.dim_embed[stage_idx] * config.mlp_ratio),
            act_layer=act_layer,
        )
        self.drop_path2 = Florence2VisionDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor):
        B, C, H, W = hidden_states.shape
        shortcut = self.conv1(hidden_states)
        shortcut = shortcut.flatten(2).transpose(1, 2)

        hidden_states = self.norm1(shortcut)
        hidden_states = hidden_states.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        hidden_states = F.pad(hidden_states, (0, 0, pad_l, pad_r, pad_t, pad_b))

        hidden_states = self.window_attn(hidden_states)
        hidden_states = shortcut + self.drop_path1(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).view(B, C, H, W)

        hidden_states = self.conv2(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.ffn(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).view(B, C, H, W)

        return hidden_states


class Florence2VisionBlock(nn.Module):
    def __init__(
        self,
        config: Florence2VisionConfig,
        stage_idx: int,
        drop_path_rate: float,
        channel_drop_path_rate: float,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.spatial_block = Florence2VisionSpatialBlock(
            config=config,
            stage_idx=stage_idx,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.channel_block = Florence2VisionChannelBlock(
            config=config,
            stage_idx=stage_idx,
            channel_drop_path_rate=channel_drop_path_rate,
        )

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.spatial_block(hidden_states)
        hidden_states = self.channel_block(hidden_states)
        return hidden_states


@auto_docstring
class Florence2VisionPreTrainedModel(PreTrainedModel):
    config_class = Florence2VisionConfig
    main_input_name = "pixel_values"
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm, nn.BatchNorm2d]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)


@auto_docstring
class Florence2VisionBackbone(Florence2VisionPreTrainedModel):
    def __init__(self, config: Florence2VisionConfig):
        super().__init__(config)
        self.config = config

        self.num_classes = config.num_classes
        self.dim_embed = config.dim_embed
        self.num_heads = config.num_heads
        self.num_groups = config.num_groups
        self.num_stages = len(self.dim_embed)

        if not (self.num_stages == len(self.num_heads) == len(self.num_groups)):
            raise ValueError(
                f"Expected self.num_stages ({self.num_stages}) == "
                f"len(self.num_heads) ({len(self.num_heads)}) == "
                f"len(self.num_groups) ({len(self.num_groups)})"
            )

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths) * 2, device="cpu")]
        depth_offset = 0

        # Resolve norm layer from config
        norm_layer = self._get_norm_layer(config.norm_layer_type)

        convs = []
        blocks = []
        for stage_idx in range(self.num_stages):
            conv_embed = Florence2VisionConvEmbed(
                config=config,
                norm_layer=norm_layer,
                stage_idx=stage_idx,
            )
            convs.append(conv_embed)

            block = nn.ModuleList(
                Florence2VisionBlock(
                    config=config,
                    stage_idx=stage_idx,
                    drop_path_rate=dpr[depth_offset + block_idx * 2],
                    channel_drop_path_rate=dpr[depth_offset + block_idx * 2 + 1],
                )
                for block_idx in range(config.depths[stage_idx])
            )
            blocks.append(block)
            depth_offset += config.depths[stage_idx] * 2

        self.convs = nn.ModuleList(convs)
        self.blocks = nn.ModuleList(blocks)
        self.norms = norm_layer(self.dim_embed[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.dim_embed[-1], self.num_classes) if self.num_classes > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def _get_norm_layer(self, norm_type: str):
        if norm_type.lower() == "layernorm":
            return nn.LayerNorm
        elif norm_type.lower() == "batchnorm":
            return nn.BatchNorm2d
        else:
            raise ValueError(f"Unsupported norm layer type: {norm_type}")

    @property
    def dim_out(self):
        return self.dim_embed[-1]

    def forward_features_unpool(self, hidden_states: torch.Tensor):
        input_size = (hidden_states.size(2), hidden_states.size(3))
        for conv, block in zip(self.convs, self.blocks):
            hidden_states, input_size = conv(hidden_states, input_size)
            for layer in block:
                hidden_states, input_size = layer(hidden_states, input_size)
        return hidden_states

    def forward_features(self, hidden_states: torch.Tensor):
        hidden_states = self.forward_features_unpool(hidden_states)
        hidden_states = self.avgpool(hidden_states.transpose(1, 2))
        hidden_states = torch.flatten(hidden_states, 1)
        hidden_states = self.norms(hidden_states)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.forward_features(hidden_states)
        hidden_states = self.head(hidden_states)
        return hidden_states


class Florence2LanguageScaledWordEmbedding(BartScaledWordEmbedding):
    pass


class Florence2LanguageEncoderLayer(BartEncoderLayer):
    pass


class Florence2LanguageDecoderLayer(BartDecoderLayer):
    pass


class Florence2LanguagePreTrainedModel(BartPreTrainedModel):
    config_class = Florence2LanguageConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = ["encoder.version", "decoder.version"]
    _no_split_modules = [r"Florence2LanguageEncoderLayer", r"Florence2LanguageDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True


class Florence2LanguageEncoder(BartEncoder):
    pass


class Florence2LanguageDecoder(BartDecoder):
    pass


class Florence2LanguageModel(BartModel):
    pass


class Florence2LanguageForConditionalGeneration(BartForConditionalGeneration, GenerationMixin):
    pass


@dataclass
class Florence2Seq2SeqLMOutput(Seq2SeqLMOutput):
    """
    Base class for Florence-2 model's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.EncoderDecoderCache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size,
            num_image_tokens, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder
    """

    image_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


FLORENCE2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Florence2Config`] or [`Florence2VisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


class Florence2PreTrainedModel(PreTrainedModel):
    config_class = Florence2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"

    @property
    def _supports_flash_attn_2(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        Flash Attention 2 or not.
        """
        return self.language_model._supports_flash_attn_2

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm]) -> None:
        std = self.config.text_config.initializer_range
        if isinstance(module, Florence2ForConditionalGeneration):
            nn.init.normal_(module.image_projection, mean=0.0, std=std)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()


FLORENCE2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details ([]`Florence2Processor`] uses
            [`CLIPImageProcessor`] for processing images).
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class Florence2ForConditionalGeneration(Florence2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = [
        "language_model.model.shared.weight",
        "language_model.model.encoder.embed_tokens.weight",
        "language_model.model.decoder.embed_tokens.weight",
    ]

    def __init__(self, config: Florence2Config):
        super().__init__(config)
        self.vision_tower = Florence2VisionBackbone(config=config.vision_config)
        # remove unused layers
        del self.vision_tower.head
        del self.vision_tower.norms

        self.vocab_size = config.vocab_size
        self._attn_implementation = config._attn_implementation
        self._build_image_projection_layers(config)

        self.language_model = Florence2LanguageForConditionalGeneration(config=config.text_config)

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    def _build_image_projection_layers(self, config: Florence2Config):
        image_dim_out = config.vision_config.dim_embed[-1]
        dim_projection = config.vision_config.projection_dim
        self.image_projection = nn.Parameter(torch.ones(image_dim_out, dim_projection))
        self.image_proj_norm = nn.LayerNorm(dim_projection)
        image_pos_embed_config = config.vision_config.image_pos_embed
        if image_pos_embed_config["type"] == "learned_abs_2d":
            self.image_pos_embed = Florence2VisionLearnedAbsolutePositionEmbedding2D(
                embedding_dim=image_dim_out, num_pos=image_pos_embed_config["max_pos_embeddings"]
            )
        else:
            raise NotImplementedError("Not implemented yet")

        self.image_feature_source = config.vision_config.image_feature_source

        # temporal embedding
        visual_temporal_embedding_config = config.vision_config.visual_temporal_embedding
        if visual_temporal_embedding_config["type"] == "COSINE":
            self.visual_temporal_embed = Florence2VisionPositionalEmbeddingCosine1D(
                embed_dim=image_dim_out, max_seq_len=visual_temporal_embedding_config["max_temporal_embeddings"]
            )
        else:
            raise NotImplementedError("Not implemented yet")

    def get_encoder(self) -> Florence2LanguageEncoder:
        return self.language_model.get_encoder()

    def get_decoder(self) -> Florence2LanguageDecoder:
        return self.language_model.get_decoder()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Linear:
        return self.language_model.get_output_embeddings()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _encode_image(self, pixel_values: torch.Tensor):
        if len(pixel_values.shape) == 4:
            batch_size, C, H, W = pixel_values.shape
            T = 1
            x = self.vision_tower.forward_features_unpool(pixel_values)
        else:
            raise ValueError(f"invalid image shape {pixel_values.shape}")

        if self.image_pos_embed is not None:
            x = x.view(batch_size * T, -1, x.shape[-1])
            num_tokens = x.shape[-2]
            h, w = int(num_tokens**0.5), int(num_tokens**0.5)
            if h * w != num_tokens:
                raise ValueError("only support square feature maps for now")
            x = x.view(batch_size * T, h, w, x.shape[-1])
            pos_embed = self.image_pos_embed(x)
            x = x + pos_embed
            x = x.view(batch_size, T * h * w, x.shape[-1])

        if self.visual_temporal_embed is not None:
            visual_temporal_embed = self.visual_temporal_embed(x.view(batch_size, T, -1, x.shape[-1])[:, :, 0])
            x = x.view(batch_size, T, -1, x.shape[-1]) + visual_temporal_embed.view(1, T, 1, x.shape[-1])

        x_feat_dict = {}

        spatial_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=2)
        x_feat_dict["spatial_avg_pool"] = spatial_avg_pool_x

        temporal_avg_pool_x = x.view(batch_size, T, -1, x.shape[-1]).mean(dim=1)
        x_feat_dict["temporal_avg_pool"] = temporal_avg_pool_x

        x = x.view(batch_size, T, -1, x.shape[-1])[:, -1]
        x_feat_dict["last_frame"] = x

        new_x = []
        for _image_feature_source in self.image_feature_source:
            if _image_feature_source not in x_feat_dict:
                raise ValueError("invalid image feature source: {}".format(_image_feature_source))
            new_x.append(x_feat_dict[_image_feature_source])

        x = torch.cat(new_x, dim=1)

        x = x @ self.image_projection
        x = self.image_proj_norm(x)

        return x

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, image_token_length = image_features.size()[:-1]
        device = image_features.device
        image_attention_mask = torch.ones(batch_size, image_token_length, device=device)

        # task_prefix_embeds: [batch_size, padded_context_length, hidden_size]
        # task_prefix_attention_mask: [batch_size, context_length]
        if inputs_embeds is None:
            return image_features, image_attention_mask

        task_prefix_embeds = inputs_embeds
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, task_prefix_embeds.size(1), device=device)

        # concat [image embeds, task prefix embeds]
        inputs_embeds = torch.cat([image_features, task_prefix_embeds], dim=1)
        attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)

        return inputs_embeds, attention_mask

    @replace_return_docstrings(output_type=Florence2Seq2SeqLMOutput, config_class="Florence2Config")
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Florence2Seq2SeqLMOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Florence2ForConditionalGeneration

        >>> model = Florence2ForConditionalGeneration.from_pretrained("microsoft/Florence-2-large")
        >>> processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large")

        >>> prompt = "<CAPTION>"
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=100)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "A green car parked in front of a yellow building."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_features = None
        if inputs_embeds is None:
            # 1. Extra the input embeddings
            if input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            # 2. Merge text and images
            if pixel_values is not None:
                # (batch_size, num_image_tokens, hidden_size)
                image_features = self._encode_image(pixel_values)
                inputs_embeds, attention_mask = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, attention_mask
                )

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.dtype)

        outputs = self.language_model(
            attention_mask=attention_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            if labels is not None:
                loss = outputs[0]
                logits = outputs[1].float()
                return (
                    loss,
                    logits,
                ) + outputs[2:]
            else:
                return outputs

        loss = None
        logits = outputs.logits
        if labels is not None:
            loss = outputs.loss
            logits = logits.float()

        return Florence2Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            image_hidden_states=image_features,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            # 1. Extra the input embeddings
            if input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            # 2. Merge text and images
            if pixel_values is not None:
                image_features = self._encode_image(pixel_values)
                inputs_embeds, attention_mask = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, attention_mask
                )

        return self.language_model.generate(
            input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self.language_model.shift_tokens_right(labels)

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)


__all__ = [
    "Florence2ForConditionalGeneration",
    "Florence2LanguageForConditionalGeneration",
    "Florence2LanguageModel",
    "Florence2LanguagePreTrainedModel",
    "Florence2PreTrainedModel",
    "Florence2VisionBackbone",
    "Florence2VisionPreTrainedModel",
]
