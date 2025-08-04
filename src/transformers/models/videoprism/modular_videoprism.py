from typing import Callable, Optional, Union
from ...utils import logging
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...utils import auto_docstring, logging, ModelOutput
from ..vivit.modeling_vivit import VivitPreTrainedModel, VivitEncoder, VivitEmbeddings, VivitLayer, VivitTubeletEmbeddings
from ..vivit.configuration_vivit import VivitConfig
from dataclasses import dataclass

logger = logging.get_logger(__name__)


class VideoPrismConfig(VivitConfig):
    def __init__(
        self,
        image_size=288,
        num_frames=16,
        tubelet_size=[1, 18, 18],
        num_channels=3,
        hidden_size=768,
        num_spatial_layers=12,
        num_temporal_layers=4,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu_python",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-06,
        qkv_bias=True,
        _attn_implementation="eager",
        atten_logit_cap=50.0,
        **kwargs,
        ):
        super().__init__()
        del self.num_hidden_layers
        self.num_spatial_layers=num_spatial_layers
        self.num_temporal_layers=num_temporal_layers
        self._attn_implementation = _attn_implementation
        self.atten_logit_cap = atten_logit_cap


def lecun_normal_(tensor):
    fan_in = tensor.size(1)  # For Embedding: (num_embeddings, embedding_dim)
    std = math.sqrt(1.0 / fan_in)
    with torch.no_grad():
        return tensor.normal_(0, std)

@dataclass
class BaseModelOutputWithSpatialAndTemporalStates(ModelOutput):
    """
    Base class for model outputs with spatial and temporal states.
    """
    last_hidden_state: Optional[torch.FloatTensor] = None
    temporal_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    spatial_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    temporal_attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    spatial_attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class VideoPrismTubeletEmbeddings(VivitTubeletEmbeddings):
    def __init__(self, config):
        super().__init__(config)

        self.image_size = config.image_size if isinstance(config.image_size, tuple) else (config.image_size, config.image_size)
        self.num_patches = (
            (self.image_size[1] // self.patch_size[2])
            * (self.image_size[0] // self.patch_size[1])
            * (self.num_frames // self.patch_size[0])
        )

    def forward(self, pixel_values, interpolate_pos_encoding: bool = False, mode='spatial'):
        
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        if not interpolate_pos_encoding and (height != self.image_size[0] or width != self.image_size[1]):
            raise ValueError(
                f"Image image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        
        x = self.projection(pixel_values) #? (B, 768, 16, 16, 16)
        
        #? I need to reshape it to (B * T, 256, 768) where 256 is the number of patches and 768 is the embedding dimension
        
        x = x.flatten(3).permute(0, 2, 3, 1)  #? (B, T, 256, 768)
        
        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  #? (B * T, 256, 768) where 256 is the number of patches and 768 is the embedding dimension

        return x


class VideoPrismEmbeddings(VivitEmbeddings):
    def __init__(self, config: VideoPrismConfig, mode:str = 'spatial'):
        super().__init__(config)
        del self.cls_token
        del self.position_embeddings
        del self.patch_embeddings

        self.mode = mode
        self.tubelet_size = config.tubelet_size
        self.pos_emb_shape = [config.num_frames, config.image_size // self.patch_size[0], config.image_size // self.patch_size[1]] #? [16, 16, 16]

        if self.mode == 'spatial':
            self.patch_embeddings = VideoPrismTubeletEmbeddings(config)
            self.spatial_pos_emb = nn.Parameter(torch.zeros(1, self.pos_emb_shape[1] * self.pos_emb_shape[2], config.hidden_size))  #? takes in patches of shape (B * T, 256, 768) returns (1, 256, 768) where 256 is the number of patches and 768 is the embedding dimension
        elif self.mode == 'temporal':
            self.temporal_pos_emb = nn.Parameter(torch.zeros(1, self.pos_emb_shape[0], config.hidden_size)) 

    def interpolate_pos_encoding(self):
        raise AttributeError("Not needed for VideoPrism")

    def forward(self, pixel_values: torch.Tensor, input_shape, interpolate_pos_encoding: bool = False):

        if self.mode == 'spatial':
            b, t, c, h, w = input_shape
            assert h == w

            embeddings = self.patch_embeddings(pixel_values)
            
            num_row_patches = h // self.tubelet_size[1]  #? 288/18 = 16
            num_column_patches = w // self.tubelet_size[2]  #? 288/18 = 16
            
            spatial_pos_emb_shape = self.pos_emb_shape[-2:]

            spatial_pos_emb = self.spatial_pos_emb  
            if spatial_pos_emb_shape != (num_row_patches, num_column_patches):  #? got a big issue here
                spatial_pos_emb = self._interpolate_emb_2d(
                    spatial_pos_emb,  #? 1, 256, 768
                    spatial_pos_emb_shape,
                    (num_row_patches, num_column_patches),
                )
                #raise ValueError(f'Positional embedding should have batch size of 1, got {self.spatial_pos_emb.shape[0]}.')
            
            embeddings = embeddings + spatial_pos_emb
            
            return embeddings            
            
        elif self.mode == 'temporal':
            if input_shape is not None:
                b, t, c, h, w = input_shape

            _, features, dim = pixel_values.shape  #? pixel_values has shape (B * T, 256, 768) where 256 is the number of patches and 768 is the embedding dimension
            
            embeddings = pixel_values.view(b, t, features, dim) #? embeddings has shape (B*T, 256, 768)
            embeddings = embeddings.permute(0, 2, 1, 3)
            embeddings = embeddings.view(b*features, t, dim)  #? embeddings has shape (B * 256, T=16, 768)
            
            temporal_seq_length = self.pos_emb_shape[0]  #? 16
            #? temporal_pos_emb shape is (1, 16, 768)
            temporal_pos_emb = self.temporal_pos_emb
            if temporal_seq_length != t:
                temporal_pos_emb = self._interpolate_emb_1d(self.temporal_pos_emb, t)
                #raise ValueError(f'Positional embedding should have batch size of 1, got {temporal_pos_emb.shape[0]}.') #! to remove
            embeddings = embeddings + temporal_pos_emb  #? embeddings has shape (B * 256, T=16, 768)
            return embeddings

        else:
            raise ValueError(f'Unknown mode: {self.mode}. Supported modes are: spatial, temporal.')

    def _interpolate_emb_2d(self, emb: torch.Tensor, source_emb_shape: tuple[int, int], target_emb_shape: tuple[int, int]):
        #? emb.shape is (1, 256, 768)
        if len(emb.shape) > 3 or emb.shape[0] != 1:
            raise ValueError('The shape of the embedding should be (1, H * W, D)')

        if emb.shape[-2] != source_emb_shape[0] * source_emb_shape[1]:  #? 16*16
            raise ValueError('The shape of the embedding does NOT match input specs.')

        emb_dim = emb.shape[-1]
        emb = emb.view(emb_dim, source_emb_shape[0], source_emb_shape[1]) #? 16, 16, 768, the first demsion is remove like squeeze
        emb = emb.unsqueeze(dim=0)
        target_emb = F.interpolate(
            emb,
            (target_emb_shape[0], target_emb_shape[1]),
            mode='bilinear',
            antialias=True, #? set to True by default in jax.image.resize
        )

        target_emb = target_emb.view(1, target_emb_shape[0] * target_emb_shape[1], emb_dim)
        return target_emb
    
    def _interpolate_emb_1d(self, emb: torch.Tensor, target_emb_length: int):
        """
        Interpolates the embedding to the target sequence length
        """
        emb_dim = emb.shape[-1]
        emb = emb.unsqueeze(dim=0)   #jnp.squeeze(emb, axis=0)

        target_emb = F.interpolate(
            emb,  #? add batch dimension
            (target_emb_length, emb_dim),
            mode='bilinear',
            antialias=True,  #? set to True by default in jax.image.resize used in the original implementation
        )
        target_emb =target_emb.squeeze(0).view(1, target_emb_length, emb_dim)
        return target_emb    


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
    # Attention logit capping
    attn_cap = torch.tensor(VideoPrismConfig().atten_logit_cap, dtype=attn_weights.dtype)   #! attention logit capping
    attn_weights = attn_cap * torch.tanh(attn_weights/attn_cap)                             #! is only supported in eager mode
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


class VideoPrismLayer(VivitLayer):
    """This corresponds to the EncoderBlock class in the scenic/videoprism implementation."""

    def __init__(self, config):
        super().__init__(config)
        del self.chunk_size_feed_forward
        del self.seq_len_dim


    def forward(self, hidden_states, head_mask=None, output_attentions=False):

        with torch.no_grad():
            self.layernorm_before.weight += nn.Parameter(torch.ones(768))
            self.layernorm_after.weight += nn.Parameter(torch.ones(768))

        super().forward(hidden_states, head_mask=head_mask, output_attentions=output_attentions)


class VideoPrismEncoder(VivitEncoder):
    def __init__(self, config: VideoPrismConfig, mode: str = 'spatial'):
        super().__init__(config)
        del self.layer
        if mode == 'spatial':
            self.layer = nn.ModuleList([VideoPrismLayer(config) for _ in range(config.num_spatial_layers)])
        elif mode == 'temporal':
            self.layer = nn.ModuleList([VideoPrismLayer(config) for _ in range(config.num_temporal_layers)])
        else:
            raise ValueError(f'Unknown mode: {mode}. Supported modes are: spatial, temporal.')


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

    def _init_weights(self, module):   #todo this needs the exact initialization as in the original VideoPrism implementation
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
            if module.mode == 'spatial':
                module.patch_embeddings.projection.weight.data = lecun_normal_(module.patch_embeddings.projection.weight.data)
                module.spatial_pos_emb.data.zero_()
            elif module.mode == 'temporal':
                module.temporal_pos_emb.data.zero_()


@auto_docstring
class VideoPrismModel(VideoPrismPreTrainedModel):
    def __init__(self, config: VideoPrismConfig):
        super().__init__(config)

        self.config = config

        self.spatial_embeddings = VideoPrismEmbeddings(
            config, mode="spatial"
        )  # ? spatial embeddings, takes in (B, T=16, C=3, H=288, W=288) and returns (B * T, 256, 768) where 256 is the number of patches and 768 is the embedding dimension

        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.temporal_embeddings = VideoPrismEmbeddings(config, mode="temporal")

        self.spatial_encoder = VideoPrismEncoder(config, mode="spatial")

        self.temporal_encoder = VideoPrismEncoder(config, mode="temporal")

        self.post_init()

    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        spatial_head_mask: Optional[torch.FloatTensor] = None,    #! These two
        temporal_head_mask: Optional[torch.FloatTensor] = None,   #! are new additions, needfurther work
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.FloatTensor], BaseModelOutputWithPooling]:
        """
        Forward pass of the VideoPrism model
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        spatial_head_mask = (
            self.get_head_mask(spatial_head_mask, self.config.num_spatial_layers)
            if spatial_head_mask is not None
            else None
        )

        temporal_head_mask = (
            self.get_head_mask(temporal_head_mask, self.config.num_temporal_layers)
            if temporal_head_mask is not None
            else None
        )

        input_shape = pixel_values.shape  # ? (B, T=16, C=3, H=288, W=288)

        spatial_embeds = self.spatial_embeddings(pixel_values, input_shape)  # ? embeds has shape (B * T, 256, 768)
        
        spatial_encoder_outputs = self.spatial_encoder(
            spatial_embeds,
            head_mask=spatial_head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # ? shape (B * T, 256, 768)

        spatial_sequence_output = spatial_encoder_outputs[0]
      
        with torch.no_grad():
            self.layernorm1.weight += nn.Parameter(torch.ones(768))    #! part of the original implementation, not sure why, could an erorr, but is necessay for matching the logits
        features = self.layernorm1(spatial_sequence_output)  # ? shape (B * T, 256, 768)
        
        #? spatial_features = (features,) + spatial_encoder_outputs[1:]  #! need to use

        temporal_embeds = self.temporal_embeddings(features, input_shape)  # ? shape (B * T, 256, 768)
        
        temporal_encoder_outputs = self.temporal_encoder(
            temporal_embeds,
            head_mask=spatial_head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # ? shape (B * T, 256, 768)

        temporal_sequence_output = temporal_encoder_outputs[0]
        
        with torch.no_grad():
            self.layernorm2.weight += nn.Parameter(torch.ones(768))

        features = self.layernorm2(temporal_sequence_output)  # ? shape is (256, 16, 768)

        #? temporal_features = (features,) + temporal_encoder_outputs[1:]
   
        features = features.view(
            input_shape[0], -1, *features.shape[1:]
        ).permute(0, 2, 1, 3).contiguous()  # ? reshape to (B, 256, 16, 768) then permute to (B, 16, 256, 768)
        features = features.view(
            input_shape[0], features.shape[1] * features.shape[2], -1
        )  # ? (B, 256*16, 768)
        if not return_dict:
            return (features, temporal_encoder_outputs.hidden_states, spatial_encoder_outputs.hidden_states, temporal_encoder_outputs.attentions, spatial_encoder_outputs.attentions)
        
        return BaseModelOutputWithSpatialAndTemporalStates(
            last_hidden_state=features,
            temporal_hidden_states=temporal_encoder_outputs.hidden_states,
            spatial_hidden_states=spatial_encoder_outputs.hidden_states,
            temporal_attentions=temporal_encoder_outputs.attentions,
            spatial_attentions=spatial_encoder_outputs.attentions,
        )  # ? returns (B * T, 256, 768) where 256 is the number of patches and 768 is the embedding dimension
        
__all__ = [
    "VideoPrismConfig",
    "VideoPrismModel",
    "VideoPrismPreTrainedModel",
]