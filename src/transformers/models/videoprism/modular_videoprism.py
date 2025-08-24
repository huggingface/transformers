import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...utils import ModelOutput, auto_docstring, logging
from ..vivit.configuration_vivit import VivitConfig
from ..vivit.modeling_vivit import (
    VivitEmbeddings,
    VivitEncoder,
    VivitLayer,
    VivitPreTrainedModel,
    VivitTubeletEmbeddings,
)


torch.set_printoptions(precision=6)

logger = logging.get_logger(__name__)


class VideoPrismConfig(VivitConfig):
    def __init__(
        self,
        image_size=288,
        num_frames=16,    # ? embeds are made using 16 frames for base and 8 frames for large model size 
        tubelet_size=[1, 18, 18],  
        num_channels=3,
        hidden_size=768,   # ? 1024 for large
        num_spatial_layers=12,     # ? 24
        num_temporal_layers=4,     # ? 4
        num_attention_heads=12,    # ? 16
        intermediate_size=3072,    # ? 4096
        hidden_act="gelu_python",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-06,
        qkv_bias=True,
        _attn_implementation="eager",
        atten_logit_cap=50.0,
        num_auxiliary_layers=2,
        enable_causal_atten=True,  #! only for text encoder
        num_unimodal_layers=12,
        vocabulary_size=32000,
        apply_l2_norm=True,
        **kwargs,
    ):
        super().__init__()
        del self.num_hidden_layers
        self.num_spatial_layers = num_spatial_layers
        self.num_temporal_layers = num_temporal_layers
        self._attn_implementation = _attn_implementation
        self.atten_logit_cap = atten_logit_cap
        self.num_auxiliary_layers = num_auxiliary_layers
        self.enable_causal_atten = enable_causal_atten  #! todo
        self.num_unimodal_layers = num_unimodal_layers
        self.vocabulary_size = vocabulary_size
        self.apply_l2_norm = apply_l2_norm


def lecun_normal_(tensor):
    fan_in = tensor.size(1)  # For Embedding: (num_embeddings, embedding_dim)
    std = math.sqrt(1.0 / fan_in)
    with torch.no_grad():
        return tensor.normal_(0, std)


@dataclass
class BaseModelOutputWithSpatialAndTemporalStates(ModelOutput):
    """
    Base class for model outputs that include spatial and temporal states.

    Args:
        last_hidden_state (Optional[torch.FloatTensor]):
            The last hidden state of the model, typically of shape 
            (batch_size, sequence_length, hidden_size).
        
        temporal_hidden_states (Optional[tuple[torch.FloatTensor, ...]]):
            A tuple containing the hidden states for each temporal layer, where each tensor 
            is of shape (batch_size, sequence_length, hidden_size). Useful for analyzing 
            temporal dynamics across layers.
        
        spatial_hidden_states (Optional[tuple[torch.FloatTensor, ...]]):
            A tuple containing the hidden states for each spatial layer, where each tensor 
            is of shape (batch_size, sequence_length, hidden_size). Useful for analyzing 
            spatial dynamics across layers.
        
        temporal_attentions (Optional[tuple[torch.FloatTensor, ...]]):
            A tuple containing the attention weights for each temporal layer, where each tensor 
            is of shape (batch_size, num_heads, sequence_length, sequence_length). Useful for 
            understanding temporal attention patterns.

        spatial_attentions (Optional[tuple[torch.FloatTensor, ...]]):
            A tuple containing the attention weights for each spatial layer, where each tensor 
            is of shape (batch_size, num_heads, sequence_length, sequence_length). Useful for 
            understanding spatial attention patterns.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    temporal_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    spatial_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    temporal_attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    spatial_attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class AttentionPoolingOutput(ModelOutput):
    """
    Base class for model outputs with attention pooling.
    """

    pooled_output: Optional[torch.FloatTensor] = None
    attention_weights: Optional[torch.FloatTensor] = None


@dataclass
class TextEncoderOutput(ModelOutput):
    """
    Base class for text encoder outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class VideoPrismClipOutput(ModelOutput):

    video_last_hidden_state: Optional[torch.FloatTensor] = None
    text_last_hidden_state: Optional[torch.FloatTensor] = None
    auxiliary_output: Optional[BaseModelOutput] = None
    attention_pooling_output: Optional[AttentionPoolingOutput] = None
    text_encoder_output: Optional[TextEncoderOutput] = None


class VideoPrismTubeletEmbeddings(VivitTubeletEmbeddings):
    def __init__(self, config):
        super().__init__(config)

        self.image_size = (
            config.image_size if isinstance(config.image_size, tuple) else (config.image_size, config.image_size)
        )
        self.num_patches = (
            (self.image_size[1] // self.patch_size[2])
            * (self.image_size[0] // self.patch_size[1])
            * (self.num_frames // self.patch_size[0])
        )

    def forward(self, pixel_values, interpolate_pos_encoding: bool = False, mode="spatial"):
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        
        if not interpolate_pos_encoding and (height != self.image_size[0] or width != self.image_size[1]):
            raise ValueError(
                f"Image image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )

        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)  # ? (B, C=3, T=16, H=288, W=288)
        
        x = self.projection(pixel_values)  # ? (B, dim=768, T=16, 16, 16), here 16, 16 = h // 18, w // 18

        x = x.flatten(3).permute(0, 2, 3, 1)  # ? (B, T=16, num_patches=256, dim=768)

        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # ? (B * T, 256, 768)
        
        return x


class VideoPrismEmbeddings(VivitEmbeddings):
    def __init__(self, config: VideoPrismConfig, mode: str = "spatial"):
        super().__init__(config)
        del self.cls_token
        del self.position_embeddings
        del self.patch_embeddings

        self.mode = mode
        self.tubelet_size = config.tubelet_size
        self.pos_emb_shape = [
            config.num_frames,
            config.image_size // self.patch_size[0],
            config.image_size // self.patch_size[1],
        ]  # ? [16, 16, 16]

        if self.mode == "spatial":
            self.patch_embeddings = VideoPrismTubeletEmbeddings(config)
            self.spatial_pos_emb = nn.Parameter(torch.zeros(1, self.pos_emb_shape[1] * self.pos_emb_shape[2], config.hidden_size))   # ? (1, 256, 768)
        
        elif self.mode == "temporal":
            self.temporal_pos_emb = nn.Parameter(torch.zeros(1, self.pos_emb_shape[0], config.hidden_size))  # ? (1, 16, 768)

    def interpolate_pos_encoding(self):
        raise AttributeError("Not needed for VideoPrism")

    def forward(self, pixel_values: torch.Tensor, input_shape, interpolate_pos_encoding: bool = False):
        
        if self.mode == "spatial":

            b, t, c, h, w = input_shape
            assert h == w
            embeddings = self.patch_embeddings(pixel_values)
            
            num_row_patches = h // self.tubelet_size[1]     # ? 288/18 = 16
            num_column_patches = w // self.tubelet_size[2]  # ? 288/18 = 16

            spatial_pos_emb_shape = self.pos_emb_shape[-2:] # ? (16, 16)

            spatial_pos_emb = self.spatial_pos_emb
            
            if spatial_pos_emb_shape != (num_row_patches, num_column_patches):
                spatial_pos_emb = self._interpolate_emb_2d(
                    spatial_pos_emb,                        # ? (1, 256, 768)
                    spatial_pos_emb_shape,                  # ? (16, 16)
                    (num_row_patches, num_column_patches),  # ? (h//18, w//18)
                )

            embeddings = embeddings + spatial_pos_emb       # ? (B * T, 256, 768)

            return embeddings

        elif self.mode == "temporal":

            if input_shape is not None:
                b, t, c, h, w = input_shape  # ? input shape before it was passed into VideoPrismModel

            _, features, dim = pixel_values.shape # ? pixel_values here corresponds to the hidden_states after spatial encoder output and has shape (B * T, 256, 768) 

            hidden_states = pixel_values.view(b, t, features, dim)      # ? (B*T, 256, 768) -> (B, T, 256, 768)
            hidden_states = hidden_states.permute(0, 2, 1, 3)           # ? (B, 256, T=16, 768)
            hidden_states = hidden_states.view(b * features, t, dim)    # ? (B * 256, T=16, 768)

            temporal_seq_length = self.pos_emb_shape[0]                 # ? 16
            
            temporal_pos_emb = self.temporal_pos_emb                    # ? (1, 16, 768)
            
            if t != temporal_seq_length:                                # ? if num_frames of input != num_frames in config
                temporal_pos_emb = self._interpolate_emb_1d(temporal_pos_emb, t)
                
            hidden_states = hidden_states + temporal_pos_emb            # ? (B * 256, T=16, 768)
            return hidden_states

        else:
            raise ValueError(f"Unknown mode: {self.mode}. Supported modes are: spatial, temporal.")

    def _interpolate_emb_2d(
        self, emb: torch.Tensor, source_emb_shape: tuple[int, int], target_emb_shape: tuple[int, int]
    ):
        # ? emb.shape is (1, 256, 768)
        if len(emb.shape) > 3 or emb.shape[0] != 1:
            raise ValueError("The shape of the embedding should be (1, H * W, D)")

        if emb.shape[-2] != source_emb_shape[0] * source_emb_shape[1]:  # ? 16*16
            raise ValueError("The shape of the embedding does NOT match input specs.")

        emb_dim = emb.shape[-1]
        emb = emb.view(
            emb_dim, source_emb_shape[0], source_emb_shape[1]
        )  # ? (768, 16, 16)
        
        emb = emb.unsqueeze(dim=0)
        target_emb = F.interpolate(
            emb,
            (target_emb_shape[0], target_emb_shape[1]),
            mode="bilinear",
            antialias=True,  # ? set to True by default in jax.image.resize
        )

        target_emb = target_emb.view(1, target_emb_shape[0] * target_emb_shape[1], emb_dim)  # ? (1, h//18 * w//18, 768)
        return target_emb

    def _interpolate_emb_1d(self, emb: torch.Tensor, target_emb_length: int):
        """
        Interpolates the embedding to the target sequence length
        """
        emb_dim = emb.shape[-1]
        emb = emb.view(1, emb_dim, -1)  # ? (1, 768, 16) for large model size
        # emb = emb.unsqueeze(dim=0)
        target_emb = F.interpolate(     #todo check if linear works, otherwise follow the exact method as in videoprism repo
            emb,                        # ? (1, 768, 16)
            target_emb_length,
            mode="linear",
            antialias=True,             # ? set to True by default in jax.image.resize used in the original implementation
        )
        # target_emb = target_emb.squeeze(0).view(1, target_emb_length, emb_dim)
        return target_emb


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    scale_logits_by_head_dims: bool = True,
    no_attention_logit_cap: Optional[float] = None,
    **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    scaling = scaling if scale_logits_by_head_dims else 1.0   # ? scale_logits_by_head_dims is set to False when PerDimScale is applied in VideoPrismClip's attention pooler
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    # Attention logit capping
    if not no_attention_logit_cap and module.config.atten_logit_cap > 0.0:
        attn_cap = torch.tensor(module.config.atten_logit_cap, dtype=attn_weights.dtype)  #! attention logit capping
        attn_weights = attn_cap * torch.tanh(attn_weights / attn_cap)                     #! is only supported in eager mode

    # Mask heads
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask.expand(*attn_weights.shape)

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)

    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class VideoPrismLayer(VivitLayer):

    def __init__(self, config):
        self.config = config
        super().__init__(config)
        del self.chunk_size_feed_forward
        del self.seq_len_dim

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        with torch.no_grad():
            self.layernorm_before.weight += nn.Parameter(
                torch.ones(self.config.hidden_size)
            )
            self.layernorm_after.weight += nn.Parameter(
                torch.ones(self.config.hidden_size)
            )

        super().forward(hidden_states, head_mask=head_mask, output_attentions=output_attentions)


class VideoPrismEncoder(VivitEncoder):
    def __init__(self, config: VideoPrismConfig, mode: str = "spatial"):
        super().__init__(config)
        del self.layer
        if mode == "spatial":
            self.layer = nn.ModuleList([VideoPrismLayer(config) for _ in range(config.num_spatial_layers)])
        elif mode == "temporal":
            self.layer = nn.ModuleList([VideoPrismLayer(config) for _ in range(config.num_temporal_layers)])
        elif mode == "auxiliary":
            self.layer = nn.ModuleList([VideoPrismLayer(config) for _ in range(config.num_auxiliary_layers)])
        elif mode == "unimodal":
            self.layer = nn.ModuleList([VideoPrismLayer(config) for _ in range(config.num_unimodal_layers)])
        else:
            raise ValueError(f"Unknown mode: {mode}. Supported modes are: spatial, temporal, auxiliary and unimodal.")

    def forward(
        self,
        hidden_states,
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

            layer_head_mask = head_mask if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

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


@auto_docstring
class VideoPrismPreTrainedModel(VivitPreTrainedModel):
    base_model_prefix = "videoprism"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _supports_sdpa = True
    _supports_flash_attn = False
    _supports_flex_attn = False
    _supports_attention_backend = True

    def _init_weights(
        self, module
    ):  # todo this needs the exact initialization as in the original VideoPrism implementation
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, VideoPrismEmbeddings):
            if module.mode == "spatial":
                module.patch_embeddings.projection.weight.data = lecun_normal_(
                    module.patch_embeddings.projection.weight.data
                )
                module.spatial_pos_emb.data.zero_()
            elif module.mode == "temporal":
                module.temporal_pos_emb.data.zero_()


@auto_docstring
class VideoPrismModel(VideoPrismPreTrainedModel):
    def __init__(self, config: VideoPrismConfig):
        super().__init__(config)

        self.config = config

        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.spatial_embeddings = VideoPrismEmbeddings(config, mode="spatial")

        self.temporal_embeddings = VideoPrismEmbeddings(config, mode="temporal")

        self.spatial_encoder = VideoPrismEncoder(config, mode="spatial")

        self.temporal_encoder = VideoPrismEncoder(config, mode="temporal")

        self.post_init()

    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,   # ? (B, T=16, C=3, H=288, W=288)
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,  #? unused at the moment
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.FloatTensor, ...], BaseModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        input_shape = pixel_values.shape  # ? (B, T=16, C=3, H=288, W=288)

        spatial_embeds = self.spatial_embeddings(pixel_values, input_shape)  # ? embeds has shape (B * T, 256, 768); embedding for each frame

        spatial_encoder_outputs = self.spatial_encoder(
            hidden_states=spatial_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # ? shape (B * T, 256, 768)
        spatial_sequence_output = spatial_encoder_outputs[0]

        with torch.no_grad():
            self.layernorm1.weight += nn.Parameter(
                torch.ones(self.config.hidden_size)
            )  #! part of the original implementation, not sure why, could be an erorr, but it is necessary for matching the logits
        features = self.layernorm1(spatial_sequence_output)  # ? shape (B * T, 256, 768)

        temporal_embeds = self.temporal_embeddings(features, input_shape)  # ? input shape (B * T, 256, 768) -> output shape (B * T, 256, 768)

        temporal_encoder_outputs = self.temporal_encoder(
            hidden_states=temporal_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # ? shape (B * 256, T=16, 768)

        temporal_sequence_output = temporal_encoder_outputs[0]

        with torch.no_grad():
            self.layernorm2.weight += nn.Parameter(torch.ones(self.config.hidden_size))

        features = self.layernorm2(temporal_sequence_output)  # ? shape is (256, 16, 768)

        features = (
            features.view(input_shape[0], -1, *features.shape[1:]).permute(0, 2, 1, 3).contiguous()
        )  # ? reshape to (B, 256, 16, 768) then permute to (B, 16, 256, 768)
        features = features.view(input_shape[0], features.shape[1] * features.shape[2], -1)  # ? (B, 256*16, 768)

        if not return_dict:
            return (
                features,
                temporal_encoder_outputs.hidden_states,
                spatial_encoder_outputs.hidden_states,
                temporal_encoder_outputs.attentions,
                spatial_encoder_outputs.attentions,
            )

        return BaseModelOutputWithSpatialAndTemporalStates(
            last_hidden_state=features,                       # ? returns (B, 4096, 768)
            temporal_hidden_states=temporal_encoder_outputs.hidden_states,
            spatial_hidden_states=spatial_encoder_outputs.hidden_states,
            temporal_attentions=temporal_encoder_outputs.attentions,
            spatial_attentions=spatial_encoder_outputs.attentions,
        )


def _l2_normalize(x: torch.Tensor, dim: int | Sequence[int] = -1, epsilon: float = 1e-12) -> torch.Tensor:
    """ L2 Normalization of a tensor along the specified axis. """
    
    norm = torch.sqrt(torch.sum(x**2, dim=dim, keepdims=True) + epsilon)
    return x / norm


class PerDimScale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = int(config.intermediate_size / config.num_attention_heads)
        self.per_dim_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, inputs):
        dim = inputs.shape[-1]  # ? dim is 256

        # ? original comments
        # 1.0/jax.nn.softplus(0.0) = 1.442695041. Hard code this number so that we
        # can avoid unnecessary XLA op fusion mess on TPU.

        r_softplus_0 = 1.442695041

        scale = torch.tensor(r_softplus_0 / (dim**0.5), dtype=inputs.dtype)
        softplus = nn.Softplus()(self.per_dim_scale).expand(*inputs.shape)
        scale = scale * softplus
        return inputs * scale


class VideoPrismMultiheadAttentionPoolingHead(nn.Module):   # ? same name pattern as in siglip 2 or aimv2
    def __init__(self, config: VideoPrismConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.intermediate_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob = config.attention_probs_dropout_prob
        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False
        self.per_dim_scale = PerDimScale(self.config)
        self.pooling_attention_query = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.query = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.qkv_bias)
        self.projection = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.qkv_bias)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor] = None,  # ? (B, 4096, 768)
        head_mask: Optional[torch.LongTensor] = None,
    ) -> AttentionPoolingOutput:
        
        batch_size, seq_length, hidden_size = hidden_states.shape        # ? (B, 4096, 768)
        query = self.pooling_attention_query.expand(batch_size, -1, -1)  # ? Expand to (B, 1, dim)
        query_layer = (
            self.query(query)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        query_layer = self.per_dim_scale(query_layer)  # ? scale via softplus function, head dimention-wise

        key_layer = (
            self.key(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        value_layer = (
            self.value(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        attention_interface: Callable = eager_attention_forward

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,         # ? is_causal is set to False obviously, but it can't be modified from the config
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
            scale_logits_by_head_dims=False,  # ? PerDimScale is applied, so we do not need to scale logits by head dims
            no_attention_logit_cap=True,      # ? to ensure that the attn logit cap is not applied for this
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = self.projection(context_layer)

        with torch.no_grad():
            self.layernorm.weight += nn.Parameter(torch.ones(self.config.hidden_size))

        outputs = self.layernorm(outputs)

        return AttentionPoolingOutput(
            pooled_output=outputs,  # ? (B, 1, 768)
            attention_weights=attention_probs
        )  


class PositionalEmbedding(nn.Module):
    def __init__(self, config: VideoPrismConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.min_timescale = 1
        self.max_timescale = 10000

    def forward(self, seq_length):
        position = torch.arange(seq_length, dtype=torch.float32).unsqueeze(0)  # ? (1, seq_length)
        num_timescales = self.hidden_size // 2

        log_timescale_increment = math.log(
            float(self.max_timescale) / float(self.min_timescale)  # ? log(10000/1) = ln(10000)
        ) / torch.maximum(torch.tensor(num_timescales, dtype=torch.float32) - 1, torch.tensor(1))

        inv_timescales = self.min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
        )

        scaled_time = position.unsqueeze(-1) * inv_timescales.expand(1, 1, -1)

        embs = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), dim=-1)

        return embs


class VideoPrismTextEncoder(nn.Module):
    def __init__(self, config: VideoPrismConfig):
        super().__init__()
        self.config = config
        self.config.hidden_act = "relu"    # ? change hidden_act from python_gelu to relu in order to reuse encoder, layer, attention code
        if config.enable_causal_atten:
            config.is_causal = True
        self.unimodal_encoder = VideoPrismEncoder(config, mode="unimodal")
        self.pos_embeddings = PositionalEmbedding(config)
        self.token_embeddings = nn.Embedding(config.vocabulary_size, config.hidden_size)
        self.cls_emb = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,     #todo
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> TextEncoderOutput:
        batch_size, seq_length = input_ids.shape
        hidden_states = self.token_embeddings(input_ids)  # ? input_ids = (B, 64)
        hidden_states = hidden_states * (self.config.hidden_size**0.5)  #! from original code
        
        cls_padding = torch.ones(batch_size, 1)
        input_ids = torch.cat(
            (input_ids, cls_padding), dim=1
        )  # ? concat CLS token, input_ids shape becomes (B, 65)
        attention_mask = torch.cat((attention_mask, cls_padding), dim=1) if attention_mask is not None else None
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_ids.shape, hidden_states.dtype, device=hidden_states.device
        )

        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype) + causal_attention_mask

        # ? the shape of input_embeds is (B, 64, 768)
        features = hidden_states + self.pos_embeddings(seq_length)
        cls_emb = self.cls_emb * (self.config.hidden_size**0.5)
        cls_emb = cls_emb.expand(features.shape[0], -1, -1)         # ? expand to (B, 1, 768)
        features = torch.cat((features, cls_emb), dim=1)            # ? features shape (B, 65, 768)

        unimodal_encoder_output = self.unimodal_encoder(
            features,
            head_mask=attention_mask if attention_mask is not None else None,  #!
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        features = unimodal_encoder_output[0]  # ? features shape (B, 65, 768)

        with torch.no_grad():
            self.layernorm.weight += nn.Parameter(torch.ones(self.config.hidden_size))

        features = self.layernorm(features)
        return TextEncoderOutput(
            last_hidden_state=features,
            hidden_states=unimodal_encoder_output.hidden_states,
            attentions=unimodal_encoder_output.attentions,
        )


class VideoPrismClip(VideoPrismPreTrainedModel):
    def __init__(self, config: VideoPrismConfig):
        super().__init__(config)
        self.config = config
        self.backbone = VideoPrismModel(config)
        self.auxiliary_encoder = VideoPrismEncoder(config, mode="auxiliary")
        self.contrastive_vision_pooler = VideoPrismMultiheadAttentionPoolingHead(config)
        self.text_encoder = VideoPrismTextEncoder(config)
        self.l2norm = _l2_normalize
        self.normalize = config.apply_l2_norm
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> BaseModelOutput:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        backbone_outputs = self.backbone(
            pixel_values=pixel_values,        # ? returns (B, 4096, 768)
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )
        
        video_features = backbone_outputs[0]
        
        auxiliary_output = self.auxiliary_encoder(
            video_features,
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, 
            return_dict=return_dict
        )                                          # ? returns (B, 4096, 768) 
        contrastive_vision_pooler_output = self.contrastive_vision_pooler(auxiliary_output[0])
        video_embeddings = contrastive_vision_pooler_output[0].squeeze(0)
        
        if self.normalize:
            video_embeddings = self.l2norm(video_embeddings, dim=-1)
        
        text_encoder_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_embeddings = text_encoder_output[0][:, -1]  # ? the cls tokens (B, 1, 768)
        
        if self.normalize:
            text_embeddings = self.l2norm(text_embeddings, dim=-1)

        return VideoPrismClipOutput(
            video_last_hidden_state=video_embeddings, 
            text_last_hidden_state=text_embeddings,
            auxiliary_output=auxiliary_output,
            attention_pooling_output=contrastive_vision_pooler_output,
            text_encoder_output=text_encoder_output,
            )


__all__ = [
    "VideoPrismConfig",
    "VideoPrismModel",
    "VideoPrismPreTrainedModel",
    "VideoPrismClip",
]
