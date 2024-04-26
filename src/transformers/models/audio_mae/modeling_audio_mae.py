# coding=utf-8
# Copyright 2024 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Audio MAE (masked autoencoder) model."""


import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_audio_mae import AudioMAEConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "AudioMAEConfig"
_CHECKPOINT_FOR_DOC = "facebook/audiomae-base"


from ..deprecated._archive_maps import VIT_MAE_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402


# Copied from transformers.models.swin.modeling_swin.window_partition
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.view(
        batch_size, height // window_size[0], window_size[0], width // window_size[1], window_size[1], num_channels
    )
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], num_channels)
    return windows


# Copied from transformers.models.swin.modeling_swin.window_reverse
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape[-1]
    windows = windows.view(-1, height // window_size[0], width // window_size[1], window_size[0], window_size[1], num_channels)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    return windows


# Copied from transformers.models.swin.modeling_swin.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output

@dataclass
# Copied from transformers.models.vit_mae.modeling_vit_mae.ViTMAEModelOutput with ViTMAE->AudioMAE
class AudioMAEModelOutput(ModelOutput):
    """
    Class for AudioMAEModel's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# Copied from transformers.models.vit_mae.modeling_vit_mae.ViTMAEDecoderOutput with ViTMAE->AudioMAE
class AudioMAEDecoderOutput(ModelOutput):
    """
    Class for AudioMAEDecoder's outputs, with potential hidden states and attentions.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# Copied from transformers.models.vit_mae.modeling_vit_mae.ViTMAEForPreTrainingOutput with ViTMAE->AudioMAE
class AudioMAEForPreTrainingOutput(ModelOutput):
    """
    Class for AudioMAEForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            Pixel reconstruction loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# Copied from transformers.models.vit_mae.modeling_vit_mae.ViTMAEEmbeddings with ViTMAE->AudioMAE
class AudioMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = AudioMAEPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size), requires_grad=False
        )
        self.config = config
        #init_weights
        # self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1], self.patch_embeddings.patch_hw, add_cls_token=True
        )
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]
            # print(noise[:3, :3])
            # for i in range(10):
            #     print(torch.rand(batch_size, seq_length, device=sequence.device)[:3, :3])
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_masked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_masked, mask, ids_restore
    
    def random_masking_2d(self, sequence, noise=None):
    
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random noise.
        2D: Spectrogram (masking t and f under mask_t_prob and mask_f_prob)

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape

        if noise is None:
            noise_t = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]
            noise_f = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]
          
        T, F = (64, 8)

        len_keep_t = int(T * (1 - self.config.mask_t_prob))
        len_keep_f = int(F * (1 - self.config.mask_f_prob))

        # noise for mask in time
        ids_shuffle_t = torch.argsort(noise_t, dim=1) # ascend: small is keep, large is remove
        ids_restore_t = torch.argsort(ids_shuffle_t, dim=1) 
        ids_keep_t = ids_shuffle_t[:,:len_keep_t]

        # noise mask in freq
        ids_shuffle_f = torch.argsort(noise_f, dim=1) # ascend: small is keep, large is remove
        ids_restore_f = torch.argsort(ids_shuffle_f, dim=1) 
        ids_keep_f = ids_shuffle_f[:,:len_keep_f] 

        # generate the binary mask: 0 is keep, 1 is remove
        # mask in freq
        mask_f = torch.ones(batch_size, F, device=sequence.device)
        mask_f[:,:len_keep_f] = 0
        mask_f = torch.gather(mask_f, dim=1, index=ids_restore_f).unsqueeze(1).repeat(1,T,1)
        # mask in time
        mask_t = torch.ones(batch_size, T, device=sequence.device)
        mask_t[:,:len_keep_t] = 0
        mask_t = torch.gather(mask_t, dim=1, index=ids_restore_t).unsqueeze(1).repeat(1,F,1).permute(0,2,1)
        mask = 1-(1-mask_t)*(1-mask_f)

        id2res=torch.Tensor(list(range(seq_length*T*F))).reshape(seq_length,T,F).to(sequence.device)
        id2res = id2res + 999*mask # add a large value for masked elements
        id2res2 = torch.argsort(id2res.flatten(start_dim=1))
        ids_keep=id2res2.flatten(start_dim=1)[:,:len_keep_f*len_keep_t]
        sequence_masked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        ids_restore = torch.argsort(id2res2.flatten(start_dim=1))
        mask = mask.flatten(start_dim=1)

        return sequence_masked, mask, ids_restore

    def forward(self, pixel_values, noise=None):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)
        # print(f"embeddings shape is {embeddings.shape}\n{embeddings[0, :3, :3]}")
        # add position embeddings w/o cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]
        # print(f'pos_embed is {self.position_embeddings[0, :3, :3]}')
        # masking: length -> length * config.mask_ratio
        if self.config.mask_2d:
            embeddings, mask, ids_restore = self.random_masking_2d(embeddings, noise)
        else:
            embeddings, mask, ids_restore = self.random_masking(embeddings, noise)            
        # print(f"x after masking shape is {embeddings.shape}\n{embeddings[0, :3, :3]}")        

        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore


#DONE
class AudioMAEPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size = (config.max_length, config.num_mel_bins) 
        patch_size = config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.patch_hw = (image_size[1] // patch_size[1], image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention ViT->AudioMAE
#DONE
class AudioMAESelfAttention(nn.Module):
    def __init__(self, config: AudioMAEConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->AudioMAE
#DONE
class AudioMAESelfOutput(nn.Module):
    """
    The residual connection is defined in AudioMAELayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: AudioMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->AudioMAE
#DONE
class AudioMAEAttention(nn.Module):
    def __init__(self, config: AudioMAEConfig) -> None:
        super().__init__()
        self.self = AudioMAESelfAttention(config)
        self.output = AudioMAESelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.self(hidden_states, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate ViT->AudioMAE
#DONE
class AudioMAEIntermediate(nn.Module):
    def __init__(self, config: AudioMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.mlp_ratio*config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTOutput ViT->AudioMAE
#DONE
class AudioMAEOutput(nn.Module):
    def __init__(self, config: AudioMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.mlp_ratio*config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->AudioMAE
#DONE
class AudioMAELayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: AudioMAEConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = AudioMAEAttention(config)
        self.intermediate = AudioMAEIntermediate(config)
        self.output = AudioMAEOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in AudioMAE, layernorm is applied before self-attention
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in AudioMAE, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs
    
# TODO convert this into intermediate and output
class Swinv2Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop_probs=[0, 0]):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Identity()#nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Identity()#nn.Dropout(drop_probs[1])

    def forward(self, x):
        print(x.shape)
        print(x[:4, :2])
        x = self.fc1(x)
        print(x[:4, :2])

        x = self.act(x)
        print(x[:4, :2])

        x = self.drop1(x)
        print(x[:4, :2])

        x = self.fc2(x)
        print(x[:4, :2])

        x = self.drop2(x)
        print(x[:4, :2])

        return x
    
class Swinv2SelfAttention(nn.Module):
    def __init__(self, config, dim, window_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attention_head_size = int(dim / self.num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )
        # mlp to generate continuous relative position bias
        self.register_parameter("tau", torch.nn.Parameter(torch.ones(self.num_heads)))
        self.meta_mlp = Swinv2Mlp(2, 384, self.num_heads, nn.ReLU, drop_probs=[0.1, 0.1])
        # get relative_coords_table
        coordinates = torch.stack(torch.meshgrid([
            torch.arange(self.window_size[0], device=self.tau.device),
            torch.arange(self.window_size[1], device=self.tau.device)], indexing='ij'), dim=0).flatten(1)
        relative_coordinates = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates = relative_coordinates.permute(1, 2, 0).reshape(-1, 2).float()
        relative_coordinates_log = torch.sign(relative_coordinates) * torch.log(
            1.0 + relative_coordinates.abs())
        self.register_buffer("relative_coordinates_log", relative_coordinates_log, persistent=False)
        
        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Identity()#nn.Dropout(config.attention_probs_dropout_prob)
        
    def _relative_positional_encodings(self) -> torch.Tensor:
        """Method computes the relative positional encodings

        Returns:
            relative_position_bias (torch.Tensor): Relative positional encodings
            (1, number of heads, window size ** 2, window size ** 2)
        """
        window_area = self.window_size[0] * self.window_size[1]
        # print(f'relative_coordinates_log is {self.relative_coordinates_log[:4]}')
        relative_position_bias = self.meta_mlp(self.relative_coordinates_log)
        # print(relative_position_bias[:3, :3])
        exit()
        relative_position_bias = relative_position_bias.transpose(1, 0).reshape(
            self.num_heads, window_area, window_area
        )
        relative_position_bias = relative_position_bias.unsqueeze(0)
        return relative_position_bias
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        batch_size, dim, num_channels = hidden_states.shape
        query = self.query(hidden_states).view(batch_size, dim, self.num_heads, num_channels // self.num_heads).permute(0, 2, 1, 3)
        key = self.key(hidden_states).view(batch_size, dim, self.num_heads, num_channels // self.num_heads).permute(0, 2, 1, 3)
        value = self.key(hidden_states).view(batch_size, dim, self.num_heads, num_channels // self.num_heads).permute(0, 2, 1, 3)
        
        denom = torch.norm(query, dim=-1, keepdim=True) @ torch.norm(key, dim=-1, keepdim=True).transpose(-2, -1)
        
        attention_scores = query @ key.transpose(-2, -1) / denom.clamp(min=1e-6)

        attention_scores = attention_scores / self.tau.clamp(min=0.01).reshape(1, self.num_heads, 1, 1)
        # print(f'attention_scores is {attention_scores.shape}\n{attention_scores[0, :3, :3, :3]}')

        attention_scores = attention_scores + self._relative_positional_encodings()
        # print(f'attention_scores is {attention_scores.shape}\n{attention_scores[0, :3, :3, :3]}')
        exit()
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in Swinv2Model forward() function)
            num_win = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // num_win, num_win, self.num_heads, dim, dim
            )
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_heads, dim, dim)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous().reshape(batch_size, dim, -1)
        # print(f'self.self is {context_layer.shape}\n{context_layer[0, :3, :3]}')
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinSelfOutput with Swin->Swinv2
class Swinv2SelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Identity()#nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class Swinv2Attention(nn.Module):
    def __init__(self, config, dim, window_size, num_heads):
        super().__init__()
        self.self = Swinv2SelfAttention(
            config=config,
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
        )
        self.output = Swinv2SelfOutput(config, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinIntermediate with Swin->Swinv2
class Swinv2Intermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dropout = nn.Identity()#nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinOutput with Swin->Swinv2
class Swinv2Output(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        self.dropout = nn.Identity()#nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

# Copied from transformers.models.swin.modeling_swin.SwinDropPath with Swin->Swinv2
class Swinv2DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)

class Swinv2Layer(nn.Module):
    def __init__(self, config, dim: int, num_heads: int, input_resolution: Tuple[int, int], window_size: Tuple[int, int], shift_size: Tuple[int, int]):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        # print(shift_size)
        self.window_size, self.shift_size = self._compute_window_shift(window_size, shift_size)
        self.num_heads = num_heads
        self.attention = Swinv2Attention(
            config=config,
            dim=self.dim,
            window_size=self.window_size,
            num_heads=self.num_heads,
        )
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.drop_path_before = Swinv2DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.intermediate = Swinv2Intermediate(config, dim)
        self.output = Swinv2Output(config, dim)
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.drop_path_after = Swinv2DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()


    def _compute_window_shift(self, target_window_size, target_shift_size) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
        return window_size, shift_size

    def get_attn_mask(self, dtype):
        # print(self.shift_size)
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            height, width = self.input_resolution
            # calculate attention mask for shifted window multihead self attention
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
            height_slices = (
                slice(0, -self.window_size[0]),
                slice(-self.window_size[0], -self.shift_size[0]),
                slice(-self.shift_size[0], None),
            )
            width_slices = (
                slice(0, -self.window_size[1]),
                slice(-self.window_size[1], -self.shift_size[1]),
                slice(-self.shift_size[1], None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = self.input_resolution
        batch_size, _, channels = hidden_states.size()
        shortcut = hidden_states
        hidden_states = hidden_states.view(batch_size, height, width, channels)
        # cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size[0] * self.window_size[1], channels)
        attn_mask = self.get_attn_mask(dtype=hidden_states.dtype)
        # print(f'hidden_states_windows is {hidden_states_windows[0, :3, :3]}')
        print(self.shift_size)
        try:
            print(f'attn_mask is {attn_mask[0, :3, :3]}')
        except:
            print(attn_mask)
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)
        #TODO debug this line attention_outputs = self.attention(
        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, output_attentions=output_attentions
        )
        # print(f'attn_windows is {attention_outputs[0][0, :3, :3]}')
        attention_output = attention_outputs[0]

        attention_windows = attention_output.view(-1, self.window_size[0], self.window_size[1], channels)
        attention_windows = window_reverse(attention_windows, self.window_size, height, width)

        # reverse cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            attention_windows = torch.roll(attention_windows, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            attention_windows = attention_windows

        attention_windows = attention_windows.view(batch_size, height * width, channels)
        
        
        hidden_states = self.layernorm_before(attention_windows)
        hidden_states = shortcut + self.drop_path_before(hidden_states)

        layer_output = self.intermediate(hidden_states)
        layer_output = self.output(layer_output)
        layer_output = hidden_states + self.drop_path_after(self.layernorm_after(layer_output))

        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs

# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->AudioMAE
#DONE
class AudioMAEEncoder(nn.Module):
    def __init__(self, config: AudioMAEConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([AudioMAELayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# Copied from transformers.models.vit_mae.modeling_vit_mae.ViTMAEPreTrainedModel with ViTMAE->AudioMAE
class AudioMAEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AudioMAEConfig
    base_model_prefix = "audiomae"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    #init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


VIT_MAE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`AudioMAEConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VIT_MAE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare AudioMAE Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_MAE_START_DOCSTRING,
)
# Copied from transformers.models.vit_mae.modeling_vit_mae.ViTMAEModel with ViTMAE->AudioMAE,facebook/vit-mae-base->facebook/audiomae-base
class AudioMAEModel(AudioMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        #TODO do we need to set mask_2d is true here or somewhere else? this will be for finetuning.
        # self.config.mask_2d = True
        self.embeddings = AudioMAEEmbeddings(config)
        self.encoder = AudioMAEEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        #init_weights
        # self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=AudioMAEModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, AudioMAEModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AudioMAEModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/audiomae-base")
        >>> model = AudioMAEModel.from_pretrained("facebook/audiomae-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output, mask, ids_restore = self.embeddings(pixel_values, noise=noise)
        # print(f'masking is {mask.shape}\n{mask[0, :3]}\n')
        # print(f'shape after embed layer is {embedding_output.shape}\n{embedding_output[0, :3, :3]}\n')
        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print(f'shape after encoder is {encoder_outputs[0].shape}\n{encoder_outputs[0][0, :3, :3]}\n')
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]

        return AudioMAEModelOutput(
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

# Copied from transformers.models.vit_mae.modeling_vit_mae.ViTMAEDecoder with ViTMAE->AudioMAE
class AudioMAEDecoder(nn.Module):
    def __init__(self, config, num_patches):
        super().__init__()
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.patch_embedding_patch_hw = (config.max_length//config.patch_size, config.num_mel_bins//config.patch_size)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
        )  # fixed sin-cos embedding
        decoder_modules = []
        for index in range(config.decoder_num_hidden_layers):
            if (index % 2) == 0:
                shift_size = (0,0)
            else:
                shift_size = (2,0)
            decoder_modules.append(
                Swinv2Layer(
                    config, #TODO decoder_config or config which to pass? better to pass decoder_config, because it will help in making this code reusable.
                    dim=config.decoder_hidden_size,
                    input_resolution=config.decoder_input_resolution,
                    num_heads=config.decoder_num_attention_heads,
                    window_size= config.window_size,
                    shift_size = shift_size,
                )
            )
        self.decoder_layers = nn.ModuleList(decoder_modules)

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2 * config.num_channels, bias=True
        )  # encoder to decoder
        self.gradient_checkpointing = False
        self.config = config
        #init_weights
        # self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],  self.patch_embedding_patch_hw, add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def forward(
        self,
        hidden_states,
        ids_restore,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # embed tokens
        hidden_states = self.decoder_embed(hidden_states)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(hidden_states.shape[0], ids_restore.shape[1] + 1 - hidden_states.shape[1], 1)
        hidden_states_ = torch.cat([hidden_states[:, 1:, :], mask_tokens], dim=1)  # no cls token
        hidden_states_ = torch.gather(hidden_states_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2]))  # unshuffle
        hidden_states = torch.cat([hidden_states[:, :1, :], hidden_states_], dim=1)  # append cls token

        # add pos embed
        hidden_states = hidden_states + self.decoder_pos_embed
        # print(f'shape after pos_embed block {hidden_states.shape}\n{hidden_states[0, :3, :3]}\n')

        hidden_states = hidden_states[:, 1:, :] # remove cls token
        # print(f'shape before decoder block {hidden_states.shape}\n{hidden_states[0, :3, :3]}\n')

        # apply Swin Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    None,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, output_attentions=output_attentions)
                # if i == 1:
                    # print(f'shape after 1st decoder block {layer_outputs[0].shape}\n{layer_outputs[0][0, :3, :3]}\n')
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        return AudioMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@add_start_docstrings(
    """The AudioMAE Model transformer with the decoder on top for self-supervised pre-training.

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>

    """,
    VIT_MAE_START_DOCSTRING,
)
# Copied from transformers.models.vit_mae.modeling_vit_mae.ViTMAEForPreTraining with ViTMAE->AudioMAE,facebook/vit-mae-base->facebook/audiomae-base
class AudioMAEForPreTraining(AudioMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.audiomae = AudioMAEModel(config)
        self.decoder = AudioMAEDecoder(config, num_patches=self.audiomae.embeddings.num_patches)

        # Initialize weights and apply final processing
        #init_weights
        # self.post_init()

    def get_input_embeddings(self):
        return self.audiomae.embeddings.patch_embeddings

    def patchify(self, pixel_values):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
        batch_size, num_channels, height, width  = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height, patch_width = (height//patch_size, width//patch_size)
        patchified_pixel_values = pixel_values.reshape(batch_size, num_channels, patch_height, patch_width, patch_size, patch_size)
        patchified_pixel_values = torch.einsum('nchpwq->nhwpqc', patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(shape=(batch_size, patch_height * patch_width, patch_size**2 * num_channels))
        return patchified_pixel_values

    def unpatchify(self, patchified_pixel_values):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
        patch_size = self.config.patch_size  
        height = self.config.max_length//patch_size
        width = self.config.num_mel_bins//patch_size
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(shape=(batch_size, height, width, patch_size, patch_size, self.config.num_channels))
        patchified_pixel_values = torch.einsum('nhwpqc->nchpwq', patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(shape=(batch_size, self.config.num_channels, height * patch_size, width * patch_size))
        return pixel_values

    def forward_loss(self, pixel_values, pred, mask):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """
        target = self.patchify(pixel_values)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=AudioMAEForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, AudioMAEForPreTrainingOutput]:
        #TODO complete this example later.
        r"""
        Returns:

        Examples:
    
        ```python
        >>> from transformers import AudioMAEFeatureExtractor, AudioMAEForPreTraining
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/audiomae-base")
        >>> model = AudioMAEForPreTraining.from_pretrained("facebook/audiomae-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> loss = outputs.loss
        >>> mask = outputs.mask
        >>> ids_restore = outputs.ids_restore
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.audiomae(
            pixel_values,
            noise=noise,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask
        # # Store latent, ids_restore, and mask externally
        # torch.save(latent, '/Users/subhashp/Documents/Open-Source/AudioMAE/emb_enc_hf.pt')
        # torch.save(ids_restore, '/Users/subhashp/Documents/Open-Source/AudioMAE/ids_restore_hf.pt')
        # torch.save(mask, '/Users/subhashp/Documents/Open-Source/AudioMAE/mask_hf.pt')
        # # Load latent
        # latent = torch.load('/Users/subhashp/Documents/Open-Source/AudioMAE/emb_enc_hf.pt')
        # # Load ids_restore
        # ids_restore = torch.load('/Users/subhashp/Documents/Open-Source/AudioMAE/ids_restore_hf.pt')
        # # Load mask
        # mask = torch.load('/Users/subhashp/Documents/Open-Source/AudioMAE/mask_hf.pt')
        # print(f'shape after encoder complete is {latent.shape}\n{latent[0, 0:3, 0:3]}\n')
        decoder_outputs = self.decoder(latent, ids_restore)
        logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        print(f'shape of logits is {logits.shape}\n{logits[0, :3, :3]}')
        loss = self.forward_loss(pixel_values, logits, mask)

        if not return_dict:
            output = (logits, mask, ids_restore) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return AudioMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
