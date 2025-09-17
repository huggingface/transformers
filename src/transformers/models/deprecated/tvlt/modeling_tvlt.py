# coding=utf-8
# Copyright 2023 MURGe-Lab and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch TVLT model."""

import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ....activations import ACT2FN
from ....modeling_layers import GradientCheckpointingLayer
from ....modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from ....modeling_utils import PreTrainedModel
from ....pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ....utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_tvlt import TvltConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TvltConfig"
_CHECKPOINT_FOR_DOC = "ZinengTang/tvlt-base"


@dataclass
class TvltModelOutput(ModelOutput):
    """
    Class for TvltModel's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        last_pixel_hidden_state (`torch.FloatTensor` of shape `(batch_size, pixel_sequence_length, hidden_size)`):
            Pixel sequence of hidden-states at the output of the last layer of the model.
        last_audio_hidden_state (`torch.FloatTensor` of shape `(batch_size, audio_sequence_length, hidden_size)`):
            Audio sequence of hidden-states at the output of the last layer of the model.
        pixel_label_masks (`torch.FloatTensor` of shape `(batch_size, pixel_patch_length)`):
            Tensor indicating which pixel patches are masked (1) and which are not (0).
        audio_label_masks (`torch.FloatTensor` of shape `(batch_size, audio_patch_length)`):
            Tensor indicating which audio patches are masked (1) and which are not (0).
        pixel_ids_restore (`torch.LongTensor` of shape `(batch_size, pixel_patch_length)`):
            Tensor containing the ids permutation of pixel masking.
        audio_ids_restore (`torch.LongTensor` of shape `(batch_size, audio_patch_length)`):
            Tensor containing the ids permutation of audio masking.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    last_pixel_hidden_state: Optional[torch.FloatTensor] = None
    last_audio_hidden_state: Optional[torch.FloatTensor] = None
    pixel_label_masks: Optional[torch.LongTensor] = None
    audio_label_masks: Optional[torch.LongTensor] = None
    pixel_ids_restore: Optional[torch.LongTensor] = None
    audio_ids_restore: Optional[torch.LongTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class TvltDecoderOutput(ModelOutput):
    """
    Class for TvltDecoder's outputs, with potential hidden states and attentions.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class TvltForPreTrainingOutput(ModelOutput):
    """
    Class for TvltForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            Pixel reconstruction loss.
        matching_logits (`torch.FloatTensor` of shape `(batch_size, 1)`):
            Matching objective logits.
        pixel_logits (`torch.FloatTensor` of shape
            `(batch_size, pixel_patch_length, image_patch_size ** 3 * pixel_num_channels)`): Pixel reconstruction
            logits.
        audio_logits (`torch.FloatTensor` of shape
            `(batch_size, audio_patch_length, image_patch_size[0] * image_patch_size[1])`): Audio reconstruction
            logits.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    matching_logits: Optional[torch.FloatTensor] = None
    pixel_logits: Optional[torch.FloatTensor] = None
    audio_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


def generate_pixel_mask_noise(pixel_values, pixel_mask=None, mask_ratio=0.75):
    """Generate noise for audio masking."""

    batch_size, seq_len = pixel_values.shape[:2]
    noise = torch.rand((batch_size, seq_len), device=pixel_values.device)  # noise in [0, 1]
    len_keep = int(seq_len * (1 - mask_ratio))
    return noise, len_keep


def generate_audio_mask_noise(audio_values, audio_mask=None, mask_ratio=0.75, mask_type="patch-level", freq_len=8):
    """Generate noise for audio masking."""

    batch_size, seq_len = audio_values.shape[:2]
    if mask_type == "frame-level":
        num_time_patches = seq_len // freq_len
        noise = (
            torch.rand(batch_size, num_time_patches, device=audio_values.device)
            .unsqueeze(-1)
            .repeat(1, 1, freq_len)
            .view(batch_size, seq_len)
        )  # noise in [0, 1]
    elif mask_type == "patch-level":
        noise = torch.rand(batch_size, seq_len, device=audio_values.device)  # noise in [0, 1]
    len_keep = int(seq_len * (1 - mask_ratio))
    return noise, len_keep


def random_masking(sequence, noise, len_keep, attention_masks=None):
    """
    Perform random masking by per-sample shuffling on frame-level. Per-sample shuffling is done by argsort random
    noise. sequence: [batch_size, seq_len, hidden_dim], sequence
    """

    batch_size, seq_len, hidden_dim = sequence.shape

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    sequence_masked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, hidden_dim))

    # generate the binary mask: 0 is keep, 1 is remove
    label_masks = torch.ones([batch_size, seq_len], device=sequence.device)
    label_masks[:, :len_keep] = 0
    # unshuffle to get the binary mask
    label_masks = torch.gather(label_masks, dim=1, index=ids_restore)

    if attention_masks is not None:
        label_masks *= attention_masks
        attention_masks = torch.gather(attention_masks, dim=1, index=ids_keep)

    return sequence_masked, attention_masks, label_masks, ids_restore


class TvltPixelEmbeddings(nn.Module):
    """Construct the patch and position embeddings."""

    def __init__(self, config):
        super().__init__()

        self.patch_embeddings = TvltPixelPatchEmbeddings(config)
        self.num_patches_per_image = self.patch_embeddings.num_patches_per_image

        self.type_embed_v = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.temporal_embed = nn.Parameter(torch.zeros(1, config.num_frames, config.hidden_size))
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.num_patches_per_image, config.hidden_size))

        self.config = config

    def forward(self, pixel_values, attention_masks=None):
        # create patch embeddings
        batch_size, num_frames, num_channels, height, width = pixel_values.shape

        embeddings = self.patch_embeddings(pixel_values)
        embeddings += self.pos_embed_v.repeat(1, num_frames, 1)
        embeddings += torch.repeat_interleave(self.temporal_embed[:, :num_frames], self.num_patches_per_image, dim=1)
        embeddings += self.type_embed_v

        return embeddings, attention_masks


class TvltAudioEmbeddings(nn.Module):
    """Construct the patch and position embeddings."""

    def __init__(self, config):
        super().__init__()

        self.patch_embeddings = TvltAudioPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches

        self.type_embed_a = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.num_freq_patches = config.frequency_length // config.audio_patch_size[1]
        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.num_patches // self.num_freq_patches, config.hidden_size))
        self.freq_embed = nn.Parameter(torch.zeros(1, self.num_freq_patches, config.hidden_size))

        self.num_freq_patches = config.frequency_length // config.audio_patch_size[1]
        self.config = config

    def forward(self, audio_values, attention_masks=None):
        # create patch embeddings
        embeddings = self.patch_embeddings(audio_values)

        num_time_patches = embeddings.size(1) // self.num_freq_patches
        embeddings += self.freq_embed.repeat(1, num_time_patches, 1)
        embeddings += torch.repeat_interleave(self.pos_embed_a[:, :num_time_patches], self.num_freq_patches, dim=1)
        embeddings += self.type_embed_a

        return embeddings, attention_masks


class TvltPixelPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.image_patch_size
        num_channels, hidden_size = config.num_image_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches_per_image = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches_per_image = num_patches_per_image
        self.hidden_size = hidden_size

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )

        pixel_values = pixel_values.reshape(batch_size * num_frames, num_channels, height, width)
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        embeddings = embeddings.reshape(batch_size, num_frames * self.num_patches_per_image, self.hidden_size)

        return embeddings


class TvltAudioPatchEmbeddings(nn.Module):
    """
    This class turns `audio_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        spectrogram_length, frequency_length, patch_size = (
            config.spectrogram_length,
            config.frequency_length,
            config.audio_patch_size,
        )
        num_channels, hidden_size = config.num_audio_channels, config.hidden_size

        spectrogram_size = (spectrogram_length, frequency_length)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (spectrogram_size[1] // patch_size[1]) * (spectrogram_size[0] // patch_size[0])
        patch_shape = (spectrogram_size[0] // patch_size[0], spectrogram_size[1] // patch_size[1])
        self.spectrogram_size = spectrogram_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = audio_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height > self.spectrogram_size[0] or width != self.spectrogram_size[1]:
            raise ValueError(
                f"Input audio size ({height}*{width}) doesn't match model"
                f" ({self.spectrogram_size[0]}*{self.spectrogram_size[1]})."
            )
        embeddings = self.projection(audio_values).flatten(2).transpose(1, 2)

        return embeddings


class TvltSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class TvltSelfOutput(nn.Module):
    """
    The residual connection is defined in TvltLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: TvltConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class TvltAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = TvltSelfAttention(config)
        self.output = TvltSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class TvltIntermediate(nn.Module):
    def __init__(self, config: TvltConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class TvltOutput(nn.Module):
    def __init__(self, config: TvltConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class TvltLayer(GradientCheckpointingLayer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = TvltAttention(config)
        self.intermediate = TvltIntermediate(config)
        self.output = TvltOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViLT, layernorm is applied before self-attention
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states.to(attention_output.device)

        # in ViLT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class TvltEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([TvltLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

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


class TvltPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config: TvltConfig
    base_model_prefix = "tvlt"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

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


TVLT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`TvltConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TVLT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`TvltProcessor`]. See [`TvltProcessor.__call__`] for
            details.

        audio_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Audio values. Audio values can be obtained using [`TvltProcessor`]. See [`TvltProcessor.__call__`] for
            details.

        pixel_mask (`torch.FloatTensor` of shape `(batch_size, num_pixel_patches)`):
            Pixel masks. Pixel masks can be obtained using [`TvltProcessor`]. See [`TvltProcessor.__call__`] for
            details.

        audio_mask (`torch.FloatTensor` of shape `(batch_size, num_audio_patches)`):
            Audio masks. Audio masks can be obtained using [`TvltProcessor`]. See [`TvltProcessor.__call__`] for
            details.

        pixel_values_mixed (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Pixel values that mix positive and negative samples in Tvlt vision-audio matching. Pixel values mixed can
            be obtained using [`TvltProcessor`]. See [`TvltProcessor.__call__`] for details.

        pixel_mask_mixed (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel masks of pixel_values_mixed. Pixel masks mixed can be obtained using [`TvltProcessor`]. See
            [`TvltProcessor.__call__`] for details.

        mask_pixel (`bool`, *optional*):
            Whether to mask pixel for MAE tasks. Only set to True in TvltForPreTraining.

        mask_audio (`bool`, *optional*):
            Whether to mask audio for MAE tasks. Only set to True in TvltForPreTraining.

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
    "The bare TVLT Model transformer outputting raw hidden-states without any specific head on top.",
    TVLT_START_DOCSTRING,
)
class TvltModel(TvltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.pixel_embeddings = TvltPixelEmbeddings(config)
        self.audio_embeddings = TvltAudioEmbeddings(config)
        self.encoder = TvltEncoder(config)

        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        if config.use_mean_pooling:
            self.layernorm = None
        else:
            self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.pixel_embeddings.patch_embeddings, self.audio_embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(TVLT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TvltModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        audio_values: torch.FloatTensor,
        pixel_mask: Optional[torch.FloatTensor] = None,
        audio_mask: Optional[torch.FloatTensor] = None,
        mask_pixel: bool = False,
        mask_audio: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.FloatTensor], TvltModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import TvltProcessor, TvltModel
        >>> import numpy as np
        >>> import torch

        >>> num_frames = 8
        >>> images = list(np.random.randn(num_frames, 3, 224, 224))
        >>> audio = list(np.random.randn(10000))

        >>> processor = TvltProcessor.from_pretrained("ZinengTang/tvlt-base")
        >>> model = TvltModel.from_pretrained("ZinengTang/tvlt-base")

        >>> input_dict = processor(images, audio, sampling_rate=44100, return_tensors="pt")

        >>> outputs = model(**input_dict)
        >>> loss = outputs.loss
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pixel_embedding_output, pixel_mask = self.pixel_embeddings(pixel_values, pixel_mask)

        audio_embedding_output, audio_mask = self.audio_embeddings(audio_values, audio_mask)

        # Mask pixel if mask_pixel is True
        pixel_label_masks = None
        pixel_ids_restore = None
        if mask_pixel:
            pixel_mask_noise, pixel_len_keep = generate_pixel_mask_noise(
                pixel_embedding_output, pixel_mask=pixel_mask, mask_ratio=self.config.pixel_mask_ratio
            )
            pixel_embedding_output, pixel_mask, pixel_label_masks, pixel_ids_restore = random_masking(
                pixel_embedding_output,
                pixel_mask_noise,
                pixel_len_keep,
                attention_masks=pixel_mask,
            )

        # Mask audio if mask_audio is True
        audio_label_masks = None
        audio_ids_restore = None
        if mask_audio:
            num_freq_patches = self.config.frequency_length // self.config.audio_patch_size[1]
            audio_mask_noise, audio_len_keep = generate_audio_mask_noise(
                audio_embedding_output,
                audio_mask=audio_mask,
                mask_ratio=self.config.audio_mask_ratio,
                mask_type=self.config.audio_mask_type,
                freq_len=num_freq_patches,
            )
            audio_embedding_output, audio_mask, audio_label_masks, audio_ids_restore = random_masking(
                audio_embedding_output,
                audio_mask_noise,
                audio_len_keep,
                attention_masks=audio_mask,
            )

        # Prepare for encoder inputs and attention masks
        batch_size = pixel_values.size(0)
        embedding_output = torch.cat(
            [self.cls_embedding.repeat(batch_size, 1, 1), pixel_embedding_output, audio_embedding_output], 1
        )
        masked_pixel_len = pixel_embedding_output.size(1)

        attention_mask = None
        if pixel_mask is not None and audio_mask is not None:
            attention_mask = torch.cat([pixel_mask[:, :1], pixel_mask, audio_mask], 1)

        input_shape = embedding_output.size()
        extended_attention_mask = None
        if attention_mask is not None:
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        if self.layernorm is not None:
            sequence_output = self.layernorm(sequence_output)

        pixel_sequence_output = sequence_output[:, 1 : 1 + masked_pixel_len]
        audio_sequence_output = sequence_output[:, 1 + masked_pixel_len :]
        if not return_dict:
            return (
                sequence_output,
                pixel_sequence_output,
                audio_sequence_output,
                pixel_label_masks,
                audio_label_masks,
                pixel_ids_restore,
                audio_ids_restore,
            ) + encoder_outputs[1:]

        return TvltModelOutput(
            last_hidden_state=sequence_output,
            last_pixel_hidden_state=pixel_sequence_output,
            last_audio_hidden_state=audio_sequence_output,
            pixel_label_masks=pixel_label_masks,
            audio_label_masks=audio_label_masks,
            pixel_ids_restore=pixel_ids_restore,
            audio_ids_restore=audio_ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TvltDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [TvltLayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        self.layernorm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False
        self.config = config

    def forward(
        self,
        hidden_states,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # predictor projection
        logits = self.layernorm(hidden_states)

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        return TvltDecoderOutput(logits=logits, hidden_states=all_hidden_states, attentions=all_self_attentions)


@add_start_docstrings(
    "The TVLT Model transformer with the decoder on top for self-supervised pre-training.",
    TVLT_START_DOCSTRING,
)
class TvltForPreTraining(TvltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.task_matching = config.task_matching
        self.task_mae = config.task_mae
        if not (self.task_matching or self.task_mae):
            raise ValueError("Must set at least one of matching task and MAE task to true")

        self.tvlt = TvltModel(config)

        if self.task_matching:
            self.matching_head = TvltMatchingHead(config)

        if self.task_mae:
            self.encoder_to_decoder = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)

            self.pixel_mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
            self.audio_mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))

            self.decoder = TvltDecoder(config)

            decoder_hidden_size = config.decoder_hidden_size

            num_frames = config.num_frames
            num_patches_per_image = self.tvlt.pixel_embeddings.num_patches_per_image
            self.decoder_pixel_pos_embed = nn.Parameter(torch.zeros(1, num_patches_per_image, decoder_hidden_size))
            self.decoder_temporal_embed = nn.Parameter(torch.zeros(1, config.num_frames, decoder_hidden_size))
            self.decoder_pixel_type_embed = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))

            num_audio_patches = self.tvlt.audio_embeddings.num_patches
            num_freq_patches = config.frequency_length // config.audio_patch_size[1]
            self.decoder_audio_pos_embed = nn.Parameter(
                torch.zeros(1, num_audio_patches // num_freq_patches, decoder_hidden_size)
            )
            self.decoder_freq_embed = nn.Parameter(torch.zeros(1, num_freq_patches, decoder_hidden_size))
            self.decoder_audio_type_embed = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))

            pixel_mae_output_dim = self.config.image_patch_size[0] ** 2 * self.config.num_image_channels
            self.pixel_mae_head = TvltMAEHead(config, pixel_mae_output_dim)
            audio_mae_output_dim = (
                self.config.audio_patch_size[0] * self.config.audio_patch_size[1] * self.config.num_audio_channels
            )
            self.audio_mae_head = TvltMAEHead(config, audio_mae_output_dim)

            self.num_frames = num_frames
            self.num_patches_per_image = num_patches_per_image
            self.num_freq_patches = num_freq_patches
            self.image_patch_size = config.image_patch_size
            self.audio_patch_size = config.audio_patch_size

        # Initialize weights and apply final processing
        self.post_init()

    def patchify_pixel(self, pixel_values):
        """
        pixel_values: [batch_size, num_frames, 3, height, width]
        """
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        num_patches_height = pixel_values.shape[3] // self.image_patch_size[0]
        num_patches_width = pixel_values.shape[4] // self.image_patch_size[1]
        patchified_pixel_values = pixel_values.reshape(
            shape=(
                batch_size,
                num_frames,
                num_channels,
                num_patches_height,
                self.image_patch_size[0],
                num_patches_width,
                self.image_patch_size[1],
            )
        )
        patchified_pixel_values = torch.einsum("ntchpwq->nthwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(
            shape=(
                batch_size,
                num_patches_height * num_patches_width * num_frames,
                self.image_patch_size[0] * self.image_patch_size[1] * num_channels,
            )
        )
        return patchified_pixel_values

    def patchify_audio(self, audio_values):
        """
        audio_values: [batch_size, 1, height, width]
        """
        batch_size, num_channels, height, width = audio_values.shape
        num_patches_height = height // self.audio_patch_size[0]
        num_patches_width = width // self.audio_patch_size[1]
        patchified_audio_values = audio_values.reshape(
            shape=(
                batch_size,
                num_channels,
                num_patches_height,
                self.audio_patch_size[0],
                num_patches_width,
                self.audio_patch_size[1],
            )
        )
        patchified_audio_values = torch.einsum("nchpwq->nhwpqc", patchified_audio_values)
        patchified_audio_values = patchified_audio_values.reshape(
            shape=(
                batch_size,
                num_patches_height * num_patches_width,
                self.audio_patch_size[0] * self.audio_patch_size[1] * num_channels,
            )
        )
        return patchified_audio_values

    def pixel_mae_loss(self, pixel_values, pixel_predictions, mask):
        patchified_pixel_values = self.patchify_pixel(pixel_values)
        loss = (pixel_predictions - patchified_pixel_values) ** 2
        loss = loss.mean(dim=-1)  # [batch_size, pixel_pixel_length], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def audio_mae_loss(self, audio_values, audio_predictions, mask):
        patchified_audio_values = self.patchify_audio(audio_values)
        loss = (audio_predictions - patchified_audio_values) ** 2
        loss = loss.mean(dim=-1)  # [batch_size, audio_pixel_length], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def concatenate_mask(self, mask_token, sequence, ids_restore):
        batch_size, seq_length, dim = sequence.shape
        mask_tokens = mask_token.repeat(batch_size, ids_restore.shape[1] - seq_length, 1)
        padded_sequence = torch.cat([sequence, mask_tokens], dim=1)
        padded_sequence = torch.gather(
            padded_sequence, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, dim)
        )  # unshuffle
        return padded_sequence

    @add_start_docstrings_to_model_forward(TVLT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TvltForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        audio_values: torch.FloatTensor,
        pixel_mask: Optional[torch.FloatTensor] = None,
        audio_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values_mixed: Optional[torch.FloatTensor] = None,
        pixel_mask_mixed: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.FloatTensor], TvltForPreTrainingOutput]:
        r"""
        pixel_values_mixed (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Pixel values that mix positive and negative samples in Tvlt vision-audio matching. Audio values can be
            obtained using [`TvltProcessor`]. See [`TvltProcessor.__call__`] for details.

        pixel_mask_mixed (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel masks of pixel_values_mixed. Pixel values mixed can be obtained using [`TvltProcessor`]. See
            [`TvltProcessor.__call__`] for details.

        labels (`torch.LongTensor` of shape `(batch_size, num_labels)`, *optional*):
            Labels for computing the vision audio matching loss. Indices should be in `[0, 1]`. num_labels has to be 1.

        Return:

        Examples:

        ```python
        >>> from transformers import TvltProcessor, TvltForPreTraining
        >>> import numpy as np
        >>> import torch

        >>> num_frames = 8
        >>> images = list(np.random.randn(num_frames, 3, 224, 224))
        >>> images_mixed = list(np.random.randn(num_frames, 3, 224, 224))
        >>> audio = list(np.random.randn(10000))
        >>> processor = TvltProcessor.from_pretrained("ZinengTang/tvlt-base")
        >>> model = TvltForPreTraining.from_pretrained("ZinengTang/tvlt-base")
        >>> input_dict = processor(
        ...     images, audio, images_mixed, sampling_rate=44100, mask_pixel=True, mask_audio=True, return_tensors="pt"
        ... )

        >>> outputs = model(**input_dict)
        >>> loss = outputs.loss
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        total_loss = 0.0

        if self.task_matching:
            if labels is None:
                raise ValueError("Matching task requires labels")
            if pixel_values_mixed is None:
                raise ValueError("Matching task requires pixel_values_mixed")

            outputs = self.tvlt(
                pixel_values_mixed,
                audio_values,
                pixel_mask=pixel_mask_mixed,
                audio_mask=audio_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = outputs[0]
            matching_logits = self.matching_head(sequence_output)

            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(matching_logits.view(-1), labels.view(-1))
            total_loss += loss

        pixel_logits = None
        audio_logits = None
        if self.task_mae and self.training:
            outputs = self.tvlt(
                pixel_values,
                audio_values,
                pixel_mask=pixel_mask,
                audio_mask=audio_mask,
                mask_pixel=True,
                mask_audio=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            pixel_sequence_output = outputs.last_pixel_hidden_state if return_dict else outputs[1]
            audio_sequence_output = outputs.last_audio_hidden_state if return_dict else outputs[2]
            pixel_label_masks = outputs.pixel_label_masks if return_dict else outputs[3]
            audio_label_masks = outputs.audio_label_masks if return_dict else outputs[4]
            pixel_ids_restore = outputs.pixel_ids_restore if return_dict else outputs[5]
            audio_ids_restore = outputs.audio_ids_restore if return_dict else outputs[6]

            pixel_decoder_input = self.encoder_to_decoder(
                pixel_sequence_output
            )  # [batch_size, num_masked_pixel_patches, decoder_hidden_size]
            audio_decoder_input = self.encoder_to_decoder(
                audio_sequence_output
            )  # [batch_size, num_masked_audio_patches, decoder_hidden_size]
            num_frames = pixel_values.size(1)
            pixel_decoder_input = self.concatenate_mask(self.pixel_mask_token, pixel_decoder_input, pixel_ids_restore)
            pixel_decoder_input = pixel_decoder_input + self.decoder_pixel_pos_embed.repeat(1, num_frames, 1)
            pixel_decoder_input = pixel_decoder_input + torch.repeat_interleave(
                self.decoder_temporal_embed[:, :num_frames], self.num_patches_per_image, dim=1
            )
            pixel_decoder_input = pixel_decoder_input + self.decoder_pixel_type_embed
            pixel_decoder_outputs = self.decoder(pixel_decoder_input)
            pixel_logits = self.pixel_mae_head(pixel_decoder_outputs.logits)

            audio_decoder_input = self.concatenate_mask(self.audio_mask_token, audio_decoder_input, audio_ids_restore)
            num_time_patches = audio_decoder_input.size(1) // self.num_freq_patches
            audio_decoder_input = audio_decoder_input + self.decoder_freq_embed.repeat(1, num_time_patches, 1)
            audio_decoder_input = audio_decoder_input + torch.repeat_interleave(
                self.decoder_audio_pos_embed[:, :num_time_patches], self.num_freq_patches, dim=1
            )
            audio_decoder_input = audio_decoder_input + self.decoder_audio_type_embed
            audio_decoder_outputs = self.decoder(audio_decoder_input)
            audio_logits = self.audio_mae_head(audio_decoder_outputs.logits)

            loss = self.pixel_mae_loss(pixel_values, pixel_logits, pixel_label_masks) + self.audio_mae_loss(
                audio_values, audio_logits, audio_label_masks
            )
            total_loss += loss

        if not return_dict:
            output = (matching_logits, pixel_logits, audio_logits) + outputs[7:]
            return ((total_loss,) + output) if loss is not None else output

        return TvltForPreTrainingOutput(
            loss=total_loss,
            matching_logits=matching_logits,
            pixel_logits=pixel_logits,
            audio_logits=audio_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TvltPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class TvltMatchingHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooler = TvltPooler(config)
        self.fc = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states):
        hidden_states = self.fc(self.pooler(hidden_states))
        return hidden_states


class TvltMAEHead(nn.Module):
    def __init__(self, config, output_dim=None):
        super().__init__()
        self.config = config
        self.decoder = nn.Linear(config.decoder_hidden_size, output_dim)

    def forward(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states


@add_start_docstrings(
    """
    Tvlt Model transformer with a classifier head on top (an MLP on top of the final hidden state of the [CLS] token)
    for audiovisual classification tasks, e.g. CMU-MOSEI Sentiment Analysis and Audio to Video Retrieval.
    """,
    TVLT_START_DOCSTRING,
)
class TvltForAudioVisualClassification(TvltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.tvlt = TvltModel(config)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2, eps=config.layer_norm_eps),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.num_labels),
        )
        self.config = config

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(TVLT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        audio_values: torch.FloatTensor,
        pixel_mask: Optional[torch.FloatTensor] = None,
        audio_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[tuple[torch.FloatTensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, num_labels)`, *optional*):
            Labels for computing the audiovisual loss. Indices should be in `[0, ..., num_classes-1]` where num_classes
            refers to the number of classes in audiovisual tasks.

        Return:

        Examples:
        ```python
        >>> from transformers import TvltProcessor, TvltForAudioVisualClassification
        >>> import numpy as np
        >>> import torch

        >>> num_frames = 8
        >>> images = list(np.random.randn(num_frames, 3, 224, 224))
        >>> audio = list(np.random.randn(10000))
        >>> processor = TvltProcessor.from_pretrained("ZinengTang/tvlt-base")
        >>> model = TvltForAudioVisualClassification.from_pretrained("ZinengTang/tvlt-base")
        >>> input_dict = processor(images, audio, sampling_rate=44100, return_tensors="pt")

        >>> outputs = model(**input_dict)
        >>> loss = outputs.loss
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.tvlt(
            pixel_values,
            audio_values,
            pixel_mask=pixel_mask,
            audio_mask=audio_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0][:, 0]
        logits = self.classifier(sequence_output)  # rank value

        loss = None
        if labels is not None:
            if self.config.loss_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits, labels)
            elif self.config.loss_type == "classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[4:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["TvltModel", "TvltForPreTraining", "TvltForAudioVisualClassification", "TvltPreTrainedModel"]
