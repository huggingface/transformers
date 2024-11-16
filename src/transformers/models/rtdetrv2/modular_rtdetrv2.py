from transformers.models.rt_detr.modeling_rt_detr import (
    RTDetrDecoderLayer, RTDetrModelOutput, RTDetrObjectDetectionOutput, RTDetrConvEncoder, RTDetrHybridEncoder,
    RTDetrEncoderLayer, RTDetrConvEncoder, RTDetrMLPPredictionHead, RTDetrDecoderOutput, RTDetrDecoderLayer, RTDetrMultiheadAttention
)
from transformers import PreTrainedModel
from transformers.models.rt_detr.configuration_rt_detr import RTDetrConfig
from transformers.activations import ACT2FN
import math
import os
import warnings
from dataclasses import dataclass
from functools import lru_cache, partial, wraps
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from transformers.utils import is_torch_cuda_available
import torch
import torch.nn as nn

class RTDetrv2Config(RTDetrConfig):
    model_type = "rt_detr_v2"

    def __init__(self, 
                 decoder_n_levels=3,  # default value
                 decoder_offset_scale=0.5,  # default value
                 **kwargs):
        # init the base RTDetrConfig class with additional parameters
        super().__init__(**kwargs)
        
        # add the new attributes with the given values or defaults
        self.decoder_n_levels = decoder_n_levels
        self.decoder_offset_scale = decoder_offset_scale

MultiScaleDeformableAttention = None
def multi_scale_deformable_attention_v2(
    value: Tensor,
    value_spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    num_points_list: List[int],
    method="default",
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points = sampling_locations.shape
    value_list = (
        value.permute(0, 2, 3, 1)
        .flatten(0, 1)
        .split([height.item() * width.item() for height, width in value_spatial_shapes], dim=-1)
    )
    # sampling_offsets [8, 480, 8, 12, 2]
    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_grids = sampling_grids.split(num_points_list, dim=-2)
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, height*width, num_heads, hidden_dim
        # -> batch_size, height*width, num_heads*hidden_dim
        # -> batch_size, num_heads*hidden_dim, height*width
        # -> batch_size*num_heads, hidden_dim, height, width
        value_l_ = value_list[level_id].reshape(batch_size * num_heads, hidden_dim, height, width)
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[level_id]
        # batch_size*num_heads, hidden_dim, num_queries, num_points
        if method == "default":
            sampling_value_l_ = nn.functional.grid_sample(
                value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        elif method == "discrete":
            sampling_coord = (sampling_grid_l_ * torch.tensor([[width, height]], device=value.device) + 0.5).to(
                torch.int64
            )

            # Separate clamping for x and y coordinates
            sampling_coord_x = sampling_coord[..., 0].clamp(0, width - 1)
            sampling_coord_y = sampling_coord[..., 1].clamp(0, height - 1)

            # Combine the clamped coordinates
            sampling_coord = torch.stack([sampling_coord_x, sampling_coord_y], dim=-1)
            sampling_coord = sampling_coord.reshape(batch_size * num_heads, num_queries * num_points_list[level_id], 2)
            sampling_idx = (
                torch.arange(sampling_coord.shape[0], device=value.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l_ = value_l_[sampling_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]
            sampling_value_l_ = sampling_value_l_.permute(0, 2, 1).reshape(
                batch_size * num_heads, hidden_dim, num_queries, num_points_list[level_id]
            )
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        batch_size * num_heads, 1, num_queries, sum(num_points_list)
    )
    output = (
        (torch.concat(sampling_value_list, dim=-1) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class RTDetrv2MultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in RTDETRv2.
    """

    def __init__(self, config: RTDetrv2Config):
        super().__init__()

        kernel_loaded = MultiScaleDeformableAttention is not None
        if is_torch_cuda_available() and is_ninja_available() and not kernel_loaded:
            try:
                load_cuda_kernels()
            except Exception as e:
                logger.warning(f"Could not load the custom kernel for multi-scale deformable attention: {e}")

        if config.d_model % config.decoder_attention_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {config.d_model} and {config.decoder_attention_heads}"
            )
        dim_per_head = config.d_model // config.decoder_attention_heads
        # check if dim_per_head is power of 2
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in RTDetrMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.im2col_step = 64

        self.d_model = config.d_model
        self.n_levels = config.decoder_n_levels
        self.n_heads = config.decoder_attention_heads
        self.offset_scale = config.decoder_offset_scale
        self.n_points = config.decoder_n_points

        n_points_list = [self.n_points for _ in range(self.n_levels)]
        self.n_points_list = n_points_list
        n_points_scale = [1 / n for n in n_points_list for _ in range(n)]
        # adding the bugger to state dict
        self.register_buffer("n_points_scale", torch.tensor(n_points_scale, dtype=torch.float32))

        self.sampling_offsets = nn.Linear(config.d_model, self.n_heads * self.n_levels * self.n_points * 2)
        self.attention_weights = nn.Linear(config.d_model, self.n_heads * self.n_levels * self.n_points)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        self.disable_custom_kernels = config.disable_custom_kernels

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        default_dtype = torch.get_default_dtype()
        thetas = torch.arange(self.n_heads, dtype=torch.int64).to(default_dtype) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            # we invert the attention_mask
            value = value.masked_fill(~attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        # batch_size, num_queries, n_heads, n_levels, n_points, 2
        num_coordinates = reference_points.shape[-1]
        if num_coordinates == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif num_coordinates == 4:
            n_points_scale = self.n_points_scale.to(dtype=hidden_states.dtype).unsqueeze(-1)
            offset = sampling_offsets * n_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        if self.disable_custom_kernels:
            # PyTorch implementation
            output = multi_scale_deformable_attention_v2(
                value, spatial_shapes, sampling_locations, attention_weights, self.n_points_list
            )
        else:
            try:
                # custom kernel
                output = MultiScaleDeformableAttentionFunction.apply(
                    value,
                    spatial_shapes,
                    level_start_index,
                    sampling_locations,
                    attention_weights,
                    self.im2col_step,
                )
            except Exception:
                # PyTorch implementation
                output = multi_scale_deformable_attention_v2(
                    value, spatial_shapes, sampling_locations, attention_weights, self.n_points_list
                )
        output = self.output_proj(output)

        return output, attention_weights

# inherit and change
class RTDetrv2DecoderLayer(nn.Module):
    def __init__(self, config: RTDetrv2Config):
        super().__init__()
        # self-attention
        self.self_attn = RTDetrMultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.decoder_activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        # cross-attention
        self.encoder_attn = RTDetrv2MultiscaleDeformableAttention(config)
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        # feedforward neural networks
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(seq_len, batch, embed_dim)`.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the queries and keys in the self-attention layer.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        second_residual = hidden_states
        # Cross-Attention
        cross_attn_weights = None
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = second_residual + hidden_states

        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


class RTDetrv2PreTrainedModel(PreTrainedModel):
    config_class = RTDetrv2Config
    base_model_prefix = "rt_detr_v2"
    main_input_name = "pixel_values"
    _no_split_modules = [r"RTDetrConvEncoder", r"RTDetrEncoderLayer", r"RTDetrDecoderLayer"]

    def _init_weights(self, module):
        """Initalize the weights"""

        """initialize linear layer bias value according to a given probability value."""
        if isinstance(module, (RTDetrv2ForObjectDetection, RTDetrv2Decoder)):
            if module.class_embed is not None:
                for layer in module.class_embed:
                    prior_prob = self.config.initializer_bias_prior_prob or 1 / (self.config.num_labels + 1)
                    bias = float(-math.log((1 - prior_prob) / prior_prob))
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, bias)

            if module.bbox_embed is not None:
                for layer in module.bbox_embed:
                    nn.init.constant_(layer.layers[-1].weight, 0)
                    nn.init.constant_(layer.layers[-1].bias, 0)

        if isinstance(module, RTDetrv2Model):
            prior_prob = self.config.initializer_bias_prior_prob or 1 / (self.config.num_labels + 1)
            bias = float(-math.log((1 - prior_prob) / prior_prob))
            nn.init.xavier_uniform_(module.encoder_score_head.weight)
            nn.init.constant_(module.encoder_score_head.bias, bias)

        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        if hasattr(module, "weight_embedding") and self.config.learn_initial_query:
            nn.init.xavier_uniform_(module.weight_embedding.weight)
        if hasattr(module, "denoising_class_embed") and self.config.num_denoising > 0:
            nn.init.xavier_uniform_(module.denoising_class_embed.weight)

# Copied from transformers.models.conditional_detr.modeling_conditional_detr.inverse_sigmoid
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
## we may use better use of inherit here
class RTDetrv2Decoder(RTDetrv2PreTrainedModel):
    def __init__(self, config: RTDetrv2Config):
        super().__init__(config)

        self.dropout = config.dropout
        self.layers = nn.ModuleList([RTDetrv2DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.query_pos_head = RTDetrMLPPredictionHead(config, 4, 2 * config.d_model, config.d_model, num_layers=2)

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:
                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)` is `as_two_stage` else `(batch_size, num_queries, 2)` or , *optional*):
                Reference point in range `[0, 1]`, top-left (0,0), bottom-right (1, 1), including padding area.
            spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`, *optional*):
                Ratio of valid area in each feature level.

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()
        intermediate_logits = ()

        reference_points = F.sigmoid(reference_points)

        # https://github.com/lyuwenyu/RT-DETR/blob/94f5e16708329d2f2716426868ec89aa774af016/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py#L252
        for idx, decoder_layer in enumerate(self.layers):
            reference_points_input = reference_points.unsqueeze(2)
            position_embeddings = self.query_pos_head(reference_points)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[idx](hidden_states)
                new_reference_points = F.sigmoid(tmp + inverse_sigmoid(reference_points))
                reference_points = new_reference_points.detach()

            intermediate += (hidden_states,)
            intermediate_reference_points += (
                (new_reference_points,) if self.bbox_embed is not None else (reference_points,)
            )

            if self.class_embed is not None:
                logits = self.class_embed[idx](hidden_states)
                intermediate_logits += (logits,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)
        if self.class_embed is not None:
            intermediate_logits = torch.stack(intermediate_logits, dim=1)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_logits,
                    intermediate_reference_points,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return RTDetrDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_logits=intermediate_logits,
            intermediate_reference_points=intermediate_reference_points,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


# could make better use of inheritence
class RTDetrv2Model(RTDetrv2PreTrainedModel):
    def __init__(self, config: RTDetrv2Config):
        super().__init__(config)

        # Create backbone
        self.backbone = RTDetrConvEncoder(config)
        intermediate_channel_sizes = self.backbone.intermediate_channel_sizes

        # Create encoder input projection layers
        # https://github.com/lyuwenyu/RT-DETR/blob/94f5e16708329d2f2716426868ec89aa774af016/rtdetr_pytorch/src/zoo/rtdetr/hybrid_encoder.py#L212
        encoder_input_proj_list = []
        for in_channels in intermediate_channel_sizes:
            encoder_input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, config.encoder_hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(config.encoder_hidden_dim),
                )
            )
        self.encoder_input_proj = nn.ModuleList(encoder_input_proj_list)

        # Create encoder
        self.encoder = RTDetrHybridEncoder(config)

        # denoising part
        if config.num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                config.num_labels + 1, config.d_model, padding_idx=config.num_labels
            )

        # decoder embedding
        if config.learn_initial_query:
            self.weight_embedding = nn.Embedding(config.num_queries, config.d_model)

        # encoder head
        self.encoder_output = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
        )
        self.encoder_score_head = nn.Linear(config.d_model, config.num_labels)
        self.encoder_bbox_head = RTDetrMLPPredictionHead(config, config.d_model, config.d_model, 4, num_layers=3)

        # init encoder output anchors and valid_mask
        if config.anchor_image_size:
            self.anchors, self.valid_mask = self.generate_anchors()

        # Create decoder input projection layers
        # https://github.com/lyuwenyu/RT-DETR/blob/94f5e16708329d2f2716426868ec89aa774af016/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py#L412
        decoder_input_proj_list = []
        for in_channels in config.decoder_in_channels:
            decoder_input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, config.d_model, kernel_size=1, bias=False),
                    nn.BatchNorm2d(config.d_model, config.batch_norm_eps),
                )
            )
        num_backbone_outs = len(config.decoder_in_channels)
        if config.num_feature_levels > num_backbone_outs + 1:
            for _ in range(config.num_feature_levels - num_backbone_outs - 1):
                decoder_input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(config.d_model, config.d_model, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(config.d_model, config.batch_norm_eps),
                    )
                )
        self.decoder_input_proj = nn.ModuleList(decoder_input_proj_list)

        # decoder
        self.decoder = RTDetrv2Decoder(config)

        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(True)

    @lru_cache(maxsize=32)
    def generate_anchors(self, spatial_shapes=None, grid_size=0.05):
        # We always generate anchors in float32 to preserve equivalence between
        # dynamic and static anchor inference
        dtype = torch.float32

        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.config.anchor_image_size[0] / s), int(self.config.anchor_image_size[1] / s)]
                for s in self.config.feat_strides
            ]
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=height, dtype=dtype), torch.arange(end=width, dtype=dtype), indexing="ij"
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_wh = torch.tensor([width, height]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_wh
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**level)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, height * width, 4))
        # define the valid range for anchor coordinates
        eps = 1e-2
        anchors = torch.concat(anchors, 1)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], RTDetrModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, RTDetrv2Model
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("")
        >>> model = RTDetrv2Model.from_pretrained("")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 300, 256]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), device=device)

        features = self.backbone(pixel_values, pixel_mask)

        proj_feats = [self.encoder_input_proj[level](source) for level, (source, mask) in enumerate(features)]

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                proj_feats,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if output_hidden_states else None,
                attentions=encoder_outputs[2]
                if len(encoder_outputs) > 2
                else encoder_outputs[1]
                if output_attentions
                else None,
            )

        # Equivalent to def _get_encoder_input
        # https://github.com/lyuwenyu/RT-DETR/blob/94f5e16708329d2f2716426868ec89aa774af016/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py#L412
        sources = []
        for level, source in enumerate(encoder_outputs[0]):
            sources.append(self.decoder_input_proj[level](source))

        # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions on the final stage
        if self.config.num_feature_levels > len(sources):
            _len_sources = len(sources)
            sources.append(self.decoder_input_proj[_len_sources](encoder_outputs[0])[-1])
            for i in range(_len_sources + 1, self.config.num_feature_levels):
                sources.append(self.decoder_input_proj[i](encoder_outputs[0][-1]))

        # Prepare encoder inputs (by flattening)
        source_flatten = []
        spatial_shapes_list = []
        for level, source in enumerate(sources):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes_list.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            source_flatten.append(source)
        source_flatten = torch.cat(source_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # prepare denoising training
        if self.training and self.config.num_denoising > 0 and labels is not None:
            (
                denoising_class,
                denoising_bbox_unact,
                attention_mask,
                denoising_meta_values,
            ) = get_contrastive_denoising_training_group(
                targets=labels,
                num_classes=self.config.num_labels,
                num_queries=self.config.num_queries,
                class_embed=self.denoising_class_embed,
                num_denoising_queries=self.config.num_denoising,
                label_noise_ratio=self.config.label_noise_ratio,
                box_noise_scale=self.config.box_noise_scale,
            )
        else:
            denoising_class, denoising_bbox_unact, attention_mask, denoising_meta_values = None, None, None, None

        batch_size = len(source_flatten)
        device = source_flatten.device
        dtype = source_flatten.dtype

        # prepare input for decoder
        if self.training or self.config.anchor_image_size is None:
            # Pass spatial_shapes as tuple to make it hashable and make sure
            # lru_cache is working for generate_anchors()
            spatial_shapes_tuple = tuple(spatial_shapes_list)
            anchors, valid_mask = self.generate_anchors(spatial_shapes_tuple)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask

        anchors, valid_mask = anchors.to(device, dtype), valid_mask.to(device, dtype)

        # use the valid_mask to selectively retain values in the feature map where the mask is `True`
        memory = valid_mask.to(source_flatten.dtype) * source_flatten

        output_memory = self.encoder_output(memory)

        enc_outputs_class = self.encoder_score_head(output_memory)
        enc_outputs_coord_logits = self.encoder_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.config.num_queries, dim=1)

        reference_points_unact = enc_outputs_coord_logits.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_logits.shape[-1])
        )

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)

        enc_topk_logits = enc_outputs_class.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])
        )

        # extract region features
        if self.config.learn_initial_query:
            target = self.weight_embedding.tile([batch_size, 1, 1])
        else:
            target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        init_reference_points = reference_points_unact.detach()

        # decoder
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=attention_mask,
            reference_points=init_reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            enc_outputs = tuple(
                value
                for value in [enc_topk_logits, enc_topk_bboxes, enc_outputs_class, enc_outputs_coord_logits]
                if value is not None
            )
            dn_outputs = tuple(value if value is not None else None for value in [denoising_meta_values])
            tuple_outputs = decoder_outputs + encoder_outputs + (init_reference_points,) + enc_outputs + dn_outputs

            return tuple_outputs

        return RTDetrModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_logits=decoder_outputs.intermediate_logits,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            init_reference_points=init_reference_points,
            enc_topk_logits=enc_topk_logits,
            enc_topk_bboxes=enc_topk_bboxes,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
            denoising_meta_values=denoising_meta_values,
        )

def _get_clones(partial_module, N):
    return nn.ModuleList([partial_module() for i in range(N)])

class RTDetrv2ForObjectDetection(RTDetrv2PreTrainedModel):
    # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    _tied_weights_keys = ["bbox_embed", "class_embed"]
    # We can't initialize the model on meta device as some weights are modified during the initialization
    _no_split_modules = None

    def __init__(self, config: RTDetrv2Config):
        super().__init__(config)

        # RTDETR encoder-decoder model
        self.model = RTDetrv2Model(config)

        # Detection heads on top
        self.class_embed = partial(nn.Linear, config.d_model, config.num_labels)
        self.bbox_embed = partial(RTDetrMLPPredictionHead, config, config.d_model, config.d_model, 4, num_layers=3)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = config.decoder_layers
        if config.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
        else:
            self.class_embed = nn.ModuleList([self.class_embed() for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed() for _ in range(num_pred)])

        # hack implementation for iterative bounding box refinement
        self.model.decoder.class_embed = self.class_embed
        self.model.decoder.bbox_embed = self.bbox_embed

        # Initialize weights and apply final processing
        self.post_init()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], RTDetrObjectDetectionOutput]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import RTDetrImageProcessor, RTDetrv2ForObjectDetection
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = RTDetrImageProcessor.from_pretrained("")
        >>> model = RTDetrv2ForObjectDetection.from_pretrained("")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 300, 80]

        >>> boxes = outputs.pred_boxes
        >>> list(boxes.shape)
        [1, 300, 4]

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
        ...     0
        ... ]

        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected sofa with confidence 0.97 at location [0.14, 0.38, 640.13, 476.21]
        Detected cat with confidence 0.96 at location [343.38, 24.28, 640.14, 371.5]
        Detected cat with confidence 0.958 at location [13.23, 54.18, 318.98, 472.22]
        Detected remote with confidence 0.951 at location [40.11, 73.44, 175.96, 118.48]
        Detected remote with confidence 0.924 at location [333.73, 76.58, 369.97, 186.99]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        denoising_meta_values = (
            outputs.denoising_meta_values if return_dict else outputs[-1] if self.training else None
        )

        outputs_class = outputs.intermediate_logits if return_dict else outputs[2]
        outputs_coord = outputs.intermediate_reference_points if return_dict else outputs[3]

        if self.training and denoising_meta_values is not None:
            dn_out_coord, outputs_coord = torch.split(outputs_coord, denoising_meta_values["dn_num_split"], dim=2)
            dn_out_class, outputs_class = torch.split(outputs_class, denoising_meta_values["dn_num_split"], dim=2)

        logits = outputs_class[:, -1]
        pred_boxes = outputs_coord[:, -1]

        loss, loss_dict, auxiliary_outputs = None, None, None
        # if labels is not None:
        #     # First: create the criterion
        #     criterion = RTDetrLoss(self.config)
        #     criterion.to(self.device)
        #     # Second: compute the losses, based on outputs and labels
        #     outputs_loss = {}
        #     outputs_loss["logits"] = logits
        #     outputs_loss["pred_boxes"] = pred_boxes
        #     if self.config.auxiliary_loss:
        #         enc_topk_logits = outputs.enc_topk_logits if return_dict else outputs[-5]
        #         enc_topk_bboxes = outputs.enc_topk_bboxes if return_dict else outputs[-4]
        #         auxiliary_outputs = self._set_aux_loss(
        #             outputs_class[:, :-1].transpose(0, 1), outputs_coord[:, :-1].transpose(0, 1)
        #         )
        #         outputs_loss["auxiliary_outputs"] = auxiliary_outputs
        #         outputs_loss["auxiliary_outputs"].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))
        #         if self.training and denoising_meta_values is not None:
        #             outputs_loss["dn_auxiliary_outputs"] = self._set_aux_loss(
        #                 dn_out_class.transpose(0, 1), dn_out_coord.transpose(0, 1)
        #             )
        #             outputs_loss["denoising_meta_values"] = denoising_meta_values

        #     loss_dict = criterion(outputs_loss, labels)

        #     loss = sum(loss_dict.values())

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + (auxiliary_outputs,) + outputs
            else:
                output = (logits, pred_boxes) + outputs
            return ((loss, loss_dict) + output) if loss is not None else output

        return RTDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_logits=outputs.intermediate_logits,
            intermediate_reference_points=outputs.intermediate_reference_points,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            init_reference_points=outputs.init_reference_points,
            enc_topk_logits=outputs.enc_topk_logits,
            enc_topk_bboxes=outputs.enc_topk_bboxes,
            enc_outputs_class=outputs.enc_outputs_class,
            enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
            denoising_meta_values=outputs.denoising_meta_values,
        )