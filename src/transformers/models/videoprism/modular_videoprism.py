from ast import Num
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pdb import post_mortem
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...utils import ModelOutput, auto_docstring, logging, torch_int
from ..t5.tokenization_t5 import T5Tokenizer
from ..t5.tokenization_t5_fast import T5TokenizerFast
from ..vivit.configuration_vivit import VivitConfig
from ..vivit.modeling_vivit import (
    VivitEmbeddings,
    VivitEncoder,
    VivitLayer,
    VivitPreTrainedModel,
    VivitTubeletEmbeddings,
)
from ..llava_onevision.video_processing_llava_onevision import LlavaOnevisionVideoProcessor



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
        num_hidden_layers=12,  #! this is just a placeholder value, num_hidden_layers will be later set from num spatial/temporal etc layers
        **kwargs,
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.num_spatial_layers = num_spatial_layers
        self.num_temporal_layers = num_temporal_layers
        self._attn_implementation = _attn_implementation
        self.atten_logit_cap = atten_logit_cap
        self.num_auxiliary_layers = num_auxiliary_layers
        self.enable_causal_atten = enable_causal_atten
        self.num_unimodal_layers = num_unimodal_layers
        self.vocabulary_size = vocabulary_size
        self.apply_l2_norm = apply_l2_norm


class VideoPrismTokenizer(T5Tokenizer):

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None
    ) -> list[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`

        Args:
            token_ids_0 (`list[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`list[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `list[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            # token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1


    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None
    ) -> list[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. VIDEOPRISM does not make
        use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`list[int]`):
                List of IDs.
            token_ids_1 (`list[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `list[int]`: List of zeros.
        """

        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return len(token_ids_0 + token_ids_1) * [0]


class VideoPrismTokenizerFast(T5TokenizerFast):
    pass

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None
    ) -> list[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`

        Args:
            token_ids_0 (`list[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`list[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `list[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # token_ids_0 = token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0
        else:
            # token_ids_1 = token_ids_1 + [self.eos_token_id]
            return self.prefix_tokens + token_ids_0 + token_ids_1

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None
    ) -> list[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`list[int]`):
                List of IDs.
            token_ids_1 (`list[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `list[int]`: List of zeros.
        """

        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return len(token_ids_0 + token_ids_1) * [0]


class VideoPrismVideoProcessor(LlavaOnevisionVideoProcessor):
    resample = PILImageResampling.BICUBIC  #! PILImageResampling.LANCZOS
    size = {"height": 288, "width": 288}
    do_normalize = False


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
        
        temporal_hidden_state (Optional[torch.FloatTensor]):
            The last hidden_state of the temporal encoder, typically of shape
            (batch_size * num_patches, num_frames, hidden_size).
        
        spatial_hidden_state (Optional[torch.FloatTensor]):
            The last hidden_state of the spatial encoder, typically of shape
            (batch_size * num_frames, num_patches, hidden_size).
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    temporal_hidden_state: Optional[torch.FloatTensor] = None
    spatial_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class AttentionPoolingOutput(ModelOutput):
    """
    Base class for model outputs with attention pooling.
    """

    pooled_output: Optional[torch.FloatTensor] = None
    attention_weights: Optional[torch.FloatTensor] = None


@dataclass
class VideoPrismClipOutput(ModelOutput):
    """
    Base class for VideoPrismClip model outputs.
    """

    logits_per_video: Optional[torch.FloatTensor] = None
    logits_per_text: Optional[torch.FloatTensor] = None
    video_embeds: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None



@dataclass
class VideoPrismVideoOutput(ModelOutput):
    """
    Base class for VideoPrismVideo model outputs.
    """
    video_last_hidden_state: Optional[torch.FloatTensor] = None
    auxiliary_output: Optional[torch.FloatTensor] = None
    attention_pooling_output: Optional[torch.FloatTensor] = None


class VideoPrismTubeletEmbeddings(VivitTubeletEmbeddings):
    def __init__(self, config):
        self.config = config
        super().__init__(config)
        del self.num_patches
        self.image_size = (
            config.image_size if isinstance(self.config.image_size, tuple) else (self.config.image_size, self.config.image_size)
        )
        self.pos_emb_shape = [
            self.image_size[0] // self.patch_size[1],
            self.image_size[1] // self.patch_size[2]
        ]
        self.num_patches = self.pos_emb_shape[0] * self.pos_emb_shape[1]

    def forward(self, pixel_values_videos, interpolate_pos_encoding: bool = False):
        batch_size, num_frames, num_channels, height, width = pixel_values_videos.shape
        if not interpolate_pos_encoding and (height != self.image_size[0] or width != self.image_size[1]):   # ! need to decide on this
            raise ValueError(
                f"Image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        # permute to (batch_size, num_channels, num_frames, height, width)
        pixel_values_videos = pixel_values_videos.permute(0, 2, 1, 3, 4)  # ? (B, C=3, T=16, H=288, W=288)
        
        hidden_states = self.projection(pixel_values_videos)  # ? (B, dim=768, T=16, 16, 16), here 16, 16 = h // 18, w // 18
        # flatten the spatial part and permute to (B, T, num_patches, dim) 
        hidden_states = hidden_states.flatten(3).permute(0, 2, 3, 1)  # ? (B, T=16, num_patches=256, dim=768)
        # combine batch and time dimension
        batch_size, num_frames, num_patches, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(batch_size * num_frames, num_patches, hidden_size)  # ? (B * T, 256, 768)
        
        return hidden_states


class VideoPrismSpatialEmbeddings(VivitEmbeddings):
    """
    VideoPrism Spatial Embeddings.

    Creates embeddings from a video using VideoPrismSpatialTubeletEmbeddings and adds positional embeddings.
    """
    def __init__(self, config: VideoPrismConfig):
        super().__init__(config)
        del self.cls_token
        self.tubelet_size = config.tubelet_size
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.patch_embeddings.num_patches, config.hidden_size))   # ? (1, 256, 768)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1]
        num_positions = self.position_embeddings.shape[1]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size[0]
        new_width = width // self.patch_size[1]

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = self.position_embeddings.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)        

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bilinear",
            antialias=True,  # ? set to True by default in jax.image.resize
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self, pixel_values_videos: torch.Tensor, interpolate_pos_encoding: bool = False):
        
        b, t, c, h, w = pixel_values_videos.shape
        assert h == w, "Input image height and width must be the same"  # ! requirement from the original repo
        embeddings = self.patch_embeddings(pixel_values_videos)
        
        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)  #! fix it
        else:
            embeddings = embeddings + self.position_embeddings
        
        embeddings = self.dropout(embeddings)

        return embeddings


class VideoPrismTemporalEmbeddings(VivitEmbeddings):
    """
    VideoPrism Temporal Embeddings.

    Receives embeddings from spatial encoder, reshapes the hidden state to 
    (batch_size * num_patches, num_frames, hidden_size) and adds positional embeddings.
    """
    def __init__(self, config: VideoPrismConfig):
        super().__init__(config)
        del self.cls_token
        del self.patch_embeddings
        del self.patch_size

        self.position_embeddings = nn.Parameter(torch.zeros(1, self.config.num_frames, config.hidden_size))  # ? (1, 16, 768)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Interpolates the embedding to the target sequence length
        """
        num_patches = embeddings.shape[1]
        num_positions = self.position_embeddings.shape[1]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions:
            return self.position_embeddings

        patch_pos_embed = self.position_embeddings

        dim = embeddings.shape[-1]

        patch_pos_embed = patch_pos_embed.reshape(1, 1, -1, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        
        patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed,                   # ? (1, 768, 1, 16)
                size=(1, num_patches),
                mode="bilinear",
                antialias=True,
            )
       
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim )

        return patch_pos_embed

    def forward(self, pixel_values_videos: torch.Tensor, input_shape, interpolate_pos_encoding: bool = False):
        if input_shape is not None:
            b, t, c, h, w = input_shape  # ? input shape before it was passed into VideoPrismModel

        _, features, dim = pixel_values_videos.shape # ? pixel_values_videos here corresponds to the hidden_states after spatial encoder output and has shape (B * T, 256, 768) 

        hidden_states = pixel_values_videos.view(b, t, features, dim)      # ? (B*T, 256, 768) -> (B, T, 256, 768)
        hidden_states = hidden_states.permute(0, 2, 1, 3)           # ? (B, 256, T=16, 768)
        embeddings = hidden_states.reshape(b * features, t, dim)    # ? (B * 256, T=16, 768)
        
        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


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


class VideoPrismLayerNorm(nn.LayerNorm):
    def forward(self, hidden_states: torch.Tensor):
        return F.layer_norm(
            hidden_states, self.normalized_shape, self.weight+1, self.bias, self.eps
        )


class VideoPrismLayer(VivitLayer):

    def __init__(self, config):
        self.config = config
        super().__init__(config)
        del self.chunk_size_feed_forward
        del self.seq_len_dim
        self.layernorm_after = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_before = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)


class VideoPrismEncoder(VivitEncoder):

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None) -> BaseModelOutput:
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask if head_mask is not None else None
            hidden_states = layer_module(hidden_states, layer_head_mask)

        return BaseModelOutput(last_hidden_state=hidden_states)

@auto_docstring
class VideoPrismPreTrainedModel(VivitPreTrainedModel):
    base_model_prefix = "videoprism"
    main_input_name = "pixel_values_videos"
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


@auto_docstring
class VideoPrismModel(VideoPrismPreTrainedModel):
    def __init__(self, config: VideoPrismConfig):
        super().__init__(config)
        self.config = config
        self.layernorm1 = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.spatial_embeddings = VideoPrismSpatialEmbeddings(self.config)
        self.temporal_embeddings = VideoPrismTemporalEmbeddings(self.config)
        self.config.num_hidden_layers = config.num_spatial_layers
        self.spatial_encoder = VideoPrismEncoder(self.config)
        self.config.num_hidden_layers = config.num_temporal_layers
        self.temporal_encoder = VideoPrismEncoder(self.config)
        self.post_init()

    @auto_docstring
    def forward(
        self,
        pixel_values_videos: Optional[torch.FloatTensor] = None,   # ? (B, T=16, C=3, H=288, W=288)
        interpolate_pos_encoding: bool = False,  #! unused at the moment
    ) -> BaseModelOutputWithSpatialAndTemporalStates:


        if pixel_values_videos is None:
            raise ValueError("You have to specify pixel_values_videos")

        input_shape = pixel_values_videos.shape  # ? (B, T=16, C=3, H=288, W=288)

        spatial_embeds = self.spatial_embeddings(pixel_values_videos)  # ? embeds has shape (B * T, 256, 768); embedding for each frame
        spatial_encoder_outputs: BaseModelOutput = self.spatial_encoder(hidden_states=spatial_embeds)  # ? shape (B * T, 256, 768)
        spatial_sequence_output = spatial_encoder_outputs.last_hidden_state
        features = self.layernorm1(spatial_sequence_output)  # ? shape (B * T, 256, 768)

        temporal_embeds = self.temporal_embeddings(features, input_shape)  # ? input shape (B * T, 256, 768) -> output shape (B * T, 256, 768)
        temporal_encoder_outputs: BaseModelOutput = self.temporal_encoder(hidden_states=temporal_embeds) # ? shape (B * 256, T=16, 768)
        temporal_sequence_output = temporal_encoder_outputs.last_hidden_state
        features = self.layernorm2(temporal_sequence_output)  # ? shape is (256, 16, 768)
        _, num_frames, dim = features.shape
        features = features.view(input_shape[0], -1, num_frames, dim).permute(0, 2, 1, 3).contiguous() # ? reshape to (B, 256, 16, 768) then permute to (B, 16, 256, 768)
        _, num_frames, num_patches, dim = features.shape
        features = features.view(input_shape[0], num_frames * num_patches, -1)  # ? (B, 16*256, 768)

        return BaseModelOutputWithSpatialAndTemporalStates(
            last_hidden_state=features,         # ? returns (B, 4096, 768)
            temporal_hidden_state=temporal_sequence_output,
            spatial_hidden_state=spatial_sequence_output,
        )


# copied from transformers.models.qwen3_next.modeling_qwen3_next.l2norm
def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


class PerDimScale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = int(config.intermediate_size / config.num_attention_heads)
        self.per_dim_scale = nn.Parameter(torch.zeros(self.dim))
        r_softplus_0 = 1.442695041
        scale = torch.tensor(r_softplus_0 / (self.dim**0.5))
        softplus = nn.functional.softplus(self.per_dim_scale)
        scale = scale * softplus
        self.register_buffer("scale", scale)

    def forward(self, inputs):
        # ? original comments
        # 1.0/jax.nn.softplus(0.0) = 1.442695041. Hard code this number so that we
        # can avoid unnecessary XLA op fusion mess on TPU.
        return inputs * self.scale.expand(*inputs.shape)


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
        self.layernorm = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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

        outputs = self.projection(context_layer)

        outputs = self.layernorm(outputs)

        return AttentionPoolingOutput(
            pooled_output=outputs,  # ? (B, 1, 768)
            attention_weights=attention_probs
        )


def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64) / (dim-2)))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=torch.int64).float(), inv_freq).float()
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)

class VideoPrismTextModel(VideoPrismPreTrainedModel):
    def __init__(self, config: VideoPrismConfig):
        super().__init__(config)
        self.config = config
        self.config.hidden_act = "relu"    # ? change hidden_act from python_gelu to relu in order to reuse encoder, layer, attention code
        if self.config.enable_causal_atten:
            self.config.is_causal = True
        self.config.num_hidden_layers = config.num_unimodal_layers
        self.unimodal_encoder = VideoPrismEncoder(self.config)
        # self.pos_embeddings = PositionalEmbedding(config)
        self.token_embeddings = nn.Embedding(config.vocabulary_size, config.hidden_size)
        self.cls_emb = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.layernorm = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.normalize = config.apply_l2_norm
        self.l2norm = l2norm
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> BaseModelOutput:
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
        features = hidden_states + create_sinusoidal_positions(seq_length, self.config.hidden_size)  # self.pos_embeddings(seq_length)
        cls_emb = self.cls_emb * (self.config.hidden_size**0.5)
        cls_emb = cls_emb.expand(features.shape[0], -1, -1)         # ? expand to (B, 1, 768)
        features = torch.cat((features, cls_emb), dim=1)            # ? features shape (B, 65, 768)

        unimodal_encoder_output = self.unimodal_encoder(
            features,
            head_mask=attention_mask if attention_mask is not None else None,  #!
        )

        features = unimodal_encoder_output.last_hidden_state  # ? features shape (B, 65, 768)

        features = self.layernorm(features)     # ! can be performed on the cls token only, for efficiency

        text_embeddings = features[:, -1]  # ? the cls token (B, 1, 768)
        
        if self.normalize:
            text_embeddings = self.l2norm(text_embeddings, dim=-1)

        return BaseModelOutput(
            last_hidden_state=text_embeddings,
        )


class VideoPrismVideoModel(VideoPrismPreTrainedModel):
    def __init__(self, config: VideoPrismConfig):
        super().__init__(config)
        self.config = config
        self.backbone = VideoPrismModel(config)
        self.config.num_hidden_layers = config.num_auxiliary_layers
        self.auxiliary_encoder = VideoPrismEncoder(self.config)
        self.contrastive_vision_pooler = VideoPrismMultiheadAttentionPoolingHead(config)
        self.l2norm = l2norm
        self.normalize = config.apply_l2_norm
        self.post_init()

    def forward(
        self,
        pixel_values_videos: torch.FloatTensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ) -> BaseModelOutput:

        backbone_outputs = self.backbone(pixel_values_videos=pixel_values_videos)        # ? returns (B, 4096, 768)        
        video_features = backbone_outputs.last_hidden_state
        auxiliary_output = self.auxiliary_encoder(video_features)          # ? returns (B, 4096, 768) 
        auxiliary_output_features = auxiliary_output.last_hidden_state
        contrastive_vision_pooler_output = self.contrastive_vision_pooler(auxiliary_output_features)
        video_embeddings = contrastive_vision_pooler_output.pooled_output  # ? (B, 1, 768)
        if self.normalize:
            video_embeddings = self.l2norm(video_embeddings, dim=-1)

        return VideoPrismVideoOutput(
            video_last_hidden_state=video_embeddings, 
            auxiliary_output=auxiliary_output,
            attention_pooling_output=contrastive_vision_pooler_output,
            )


class VideoPrismClipModel(VideoPrismPreTrainedModel):   
    def __init__(self, config: VideoPrismConfig):
        super().__init__(config)
        self.config = config
        self.video_model = VideoPrismVideoModel(config)
        self.text_model = VideoPrismTextModel(config)
        self.post_init()

    def forward(
        self,
        pixel_values_videos: Optional[torch.FloatTensor] = None,   # ? (B, T=16, C=3, H=288, W=288)
        input_ids: Optional[torch.Tensor] = None,            # ? (B, 64)
        attention_mask: Optional[torch.Tensor] = None,       # ? (B, 64)
        temperature: Optional[float] = None,
    ) -> VideoPrismClipOutput:

        if pixel_values_videos is None:
            raise ValueError("You have to specify pixel_values_videos")
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        video_model_outputs = self.video_model(pixel_values_videos=pixel_values_videos)
        text_model_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)

        video_embeddings = video_model_outputs.video_last_hidden_state  # ? (video_batch, 1, 768)
        text_embeddings = text_model_outputs.last_hidden_state          # ? (text_batch, 768)
        emb_dim = video_embeddings[0].shape[-1]
        assert emb_dim == text_embeddings[0].shape[-1]

        video_embeds = video_embeddings.reshape(-1, emb_dim)
        text_embeds = text_embeddings.reshape(-1, emb_dim)
        similarity_matrix = torch.matmul(video_embeds, text_embeds.T)

        if temperature is not None:
            similarity_matrix /= temperature

        logits_per_video = torch.exp(similarity_matrix)
        logits_per_text = logits_per_video.T
        logits_per_video = logits_per_video / torch.sum(logits_per_video, dim=0, keepdims=True)
        logits_per_text = logits_per_text / torch.sum(logits_per_text, dim=0, keepdims=True)

        return VideoPrismClipOutput(
            logits_per_video=logits_per_video,
            logits_per_text=logits_per_text,
            video_embeds=video_embeds,
            text_embeds=text_embeds,

        )


__all__ = [
    "VideoPrismConfig",
    "VideoPrismModel",
    "VideoPrismPreTrainedModel",
    "VideoPrismClipModel",
    "VideoPrismTokenizer",
    "VideoPrismTokenizerFast",
    "VideoPrismVideoProcessor",
]
