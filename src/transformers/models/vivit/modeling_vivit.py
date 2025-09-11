# coding=utf-8
# Copyright 2023 Google AI and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch ViViT model."""

from typing import Callable, Optional

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import TransformersKwargs, auto_docstring, logging, torch_int
from ...utils.generic import can_return_tuple, check_model_inputs
from .configuration_vivit import VivitConfig


logger = logging.get_logger(__name__)


class VivitTubeletEmbeddings(nn.Module):
    """
    Construct Vivit Tubelet embeddings.

    This module turns a batch of videos of shape (batch_size, num_frames, num_channels, height, width) into a tensor of
    shape (batch_size, seq_len, hidden_size) to be consumed by a Transformer encoder.

    The seq_len (the number of patches) equals (number of frames // tubelet_size[0]) * (height // tubelet_size[1]) *
    (width // tubelet_size[2]).
    """

    def __init__(self, config: VivitConfig):
        super().__init__()
        self.num_frames = config.num_frames
        self.image_size = config.image_size
        self.patch_size = config.tubelet_size
        self.num_patches = (
            (self.image_size // self.patch_size[2])
            * (self.image_size // self.patch_size[1])
            * (self.num_frames // self.patch_size[0])
        )
        self.embed_dim = config.hidden_size

        self.projection = nn.Conv3d(
            config.num_channels, config.hidden_size, kernel_size=config.tubelet_size, stride=config.tubelet_size
        )

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        if not interpolate_pos_encoding and (height != self.image_size or width != self.image_size):
            raise ValueError(
                f"Image image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )

        # permute to (batch_size, num_channels, num_frames, height, width)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)

        x = self.projection(pixel_values)
        # out_batch_size, out_num_channels, out_num_frames, out_height, out_width = x.shape
        # flattens time and space dimensions, transposes to (out_batch_size, flat_tokens, out_num_channels)
        x = x.flatten(2).transpose(1, 2)
        return x


class VivitEmbeddings(nn.Module):
    """
    Vivit Embeddings.

    Creates embeddings from a video using VivitTubeletEmbeddings, adds CLS token and positional embeddings.
    """

    def __init__(self, config: VivitConfig):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = VivitTubeletEmbeddings(config)

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.patch_embeddings.num_patches + 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.tubelet_size[1:]
        self.config = config

    # Adapted from transformers.models.vit.modeling_vit.ViTEmbeddings.interpolate_pos_encoding
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size[0]
        new_width = width // self.patch_size[1]

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        cls_tokens = self.cls_token.tile([batch_size, 1, 1])
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


# Copied from transformers.models.vit.modeling_vit.eager_attention_forward
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


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->Vivit
class VivitSelfAttention(nn.Module):
    def __init__(self, config: VivitConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob = config.attention_probs_dropout_prob
        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

    def forward(
        self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        new_shape = batch_size, -1, self.num_attention_heads, self.attention_head_size

        key_layer = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*new_shape).transpose(1, 2)
        query_layer = self.query(hidden_states).view(*new_shape).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer, attention_probs


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->Vivit
class VivitSelfOutput(nn.Module):
    """
    The residual connection is defined in VivitLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: VivitConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->Vivit
class VivitAttention(nn.Module):
    def __init__(self, config: VivitConfig):
        super().__init__()
        self.attention = VivitSelfAttention(config)
        self.output = VivitSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: set[int]):
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

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self_attn_output, _ = self.attention(hidden_states, head_mask)
        output = self.output(self_attn_output, hidden_states)
        return output


class VivitIntermediate(nn.Module):
    def __init__(self, config: VivitConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class VivitOutput(nn.Module):
    def __init__(self, config: VivitConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class VivitLayer(GradientCheckpointingLayer):
    """This corresponds to the EncoderBlock class in the scenic/vivit implementation."""

    def __init__(self, config: VivitConfig):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = VivitAttention(config)
        self.intermediate = VivitIntermediate(config)
        self.output = VivitOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states_norm = self.layernorm_before(hidden_states)
        attention_output = self.attention(hidden_states_norm, head_mask)

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in Vivit, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        return layer_output


class VivitEncoder(nn.Module):
    def __init__(self, config: VivitConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([VivitLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None) -> BaseModelOutput:
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            hidden_states = layer_module(hidden_states, layer_head_mask)

        return BaseModelOutput(last_hidden_state=hidden_states)


class VivitPooler(nn.Module):
    def __init__(self, config: VivitConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@auto_docstring
class VivitPreTrainedModel(PreTrainedModel):
    config: VivitConfig
    base_model_prefix = "vivit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": VivitLayer,
        "attentions": VivitSelfAttention,
    }

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, VivitEmbeddings):
            module.cls_token.data.zero_()
            module.position_embeddings.data.zero_()


@auto_docstring
class VivitModel(VivitPreTrainedModel):
    def __init__(self, config: VivitConfig, add_pooling_layer: bool = True):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
        super().__init__(config)
        self.config = config

        self.embeddings = VivitEmbeddings(config)
        self.encoder = VivitEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = VivitPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model.

        Args:
            heads_to_prune:
                dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> import av
        >>> import numpy as np

        >>> from transformers import VivitImageProcessor, VivitModel
        >>> from huggingface_hub import hf_hub_download

        >>> np.random.seed(0)


        >>> def read_video_pyav(container, indices):
        ...     '''
        ...     Decode the video with PyAV decoder.
        ...     Args:
        ...         container (`av.container.input.InputContainer`): PyAV container.
        ...         indices (`list[int]`): List of frame indices to decode.
        ...     Returns:
        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...     '''
        ...     frames = []
        ...     container.seek(0)
        ...     start_index = indices[0]
        ...     end_index = indices[-1]
        ...     for i, frame in enumerate(container.decode(video=0)):
        ...         if i > end_index:
        ...             break
        ...         if i >= start_index and i in indices:
        ...             frames.append(frame)
        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...     '''
        ...     Sample a given number of frame indices from the video.
        ...     Args:
        ...         clip_len (`int`): Total number of frames to sample.
        ...         frame_sample_rate (`int`): Sample every n-th frame.
        ...         seg_len (`int`): Maximum allowed index of sample's last frame.
        ...     Returns:
        ...         indices (`list[int]`): List of sampled frame indices
        ...     '''
        ...     converted_len = int(clip_len * frame_sample_rate)
        ...     end_idx = np.random.randint(converted_len, seg_len)
        ...     start_idx = end_idx - converted_len
        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...     return indices


        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
        >>> file_path = hf_hub_download(
        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
        ... )
        >>> container = av.open(file_path)

        >>> # sample 32 frames
        >>> indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        >>> video = read_video_pyav(container=container, indices=indices)

        >>> image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        >>> model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")

        >>> # prepare video for the model
        >>> inputs = image_processor(list(video), return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 3137, 768]
        ```"""

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        encoder_outputs: BaseModelOutput = self.encoder(embedding_output, head_mask=head_mask)
        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output)


@auto_docstring(
    custom_intro="""
        ViViT Transformer model with a video classification head on top (a linear layer on top of the final hidden state of the
    [CLS] token) e.g. for Kinetics-400.

        <Tip>

            Note that it's possible to fine-tune ViT on higher resolution images than the ones it has been trained on, by
            setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
            position embeddings to the higher resolution.

        </Tip>
    """
)
class VivitForVideoClassification(VivitPreTrainedModel):
    def __init__(self, config: VivitConfig):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vivit = VivitModel(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Examples:

        ```python
        >>> import av
        >>> import numpy as np
        >>> import torch

        >>> from transformers import VivitImageProcessor, VivitForVideoClassification
        >>> from huggingface_hub import hf_hub_download

        >>> np.random.seed(0)


        >>> def read_video_pyav(container, indices):
        ...     '''
        ...     Decode the video with PyAV decoder.
        ...     Args:
        ...         container (`av.container.input.InputContainer`): PyAV container.
        ...         indices (`list[int]`): List of frame indices to decode.
        ...     Returns:
        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...     '''
        ...     frames = []
        ...     container.seek(0)
        ...     start_index = indices[0]
        ...     end_index = indices[-1]
        ...     for i, frame in enumerate(container.decode(video=0)):
        ...         if i > end_index:
        ...             break
        ...         if i >= start_index and i in indices:
        ...             frames.append(frame)
        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...     '''
        ...     Sample a given number of frame indices from the video.
        ...     Args:
        ...         clip_len (`int`): Total number of frames to sample.
        ...         frame_sample_rate (`int`): Sample every n-th frame.
        ...         seg_len (`int`): Maximum allowed index of sample's last frame.
        ...     Returns:
        ...         indices (`list[int]`): List of sampled frame indices
        ...     '''
        ...     converted_len = int(clip_len * frame_sample_rate)
        ...     end_idx = np.random.randint(converted_len, seg_len)
        ...     start_idx = end_idx - converted_len
        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...     return indices


        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
        >>> file_path = hf_hub_download(
        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
        ... )
        >>> container = av.open(file_path)

        >>> # sample 32 frames
        >>> indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
        >>> video = read_video_pyav(container=container, indices=indices)

        >>> image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        >>> model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")

        >>> inputs = image_processor(list(video), return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ...     logits = outputs.logits

        >>> # model predicts one of the 400 Kinetics-400 classes
        >>> predicted_label = logits.argmax(-1).item()
        >>> print(model.config.id2label[predicted_label])
        LABEL_116
        ```"""

        outputs: BaseModelOutput = self.vivit(
            pixel_values, head_mask=head_mask, interpolate_pos_encoding=interpolate_pos_encoding, **kwargs
        )
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config, **kwargs)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["VivitModel", "VivitPreTrainedModel", "VivitForVideoClassification"]
