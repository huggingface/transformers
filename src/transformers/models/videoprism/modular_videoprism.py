
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...processing_utils import Unpack
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...utils import ModelOutput, auto_docstring, logging, torch_int, TransformersKwargs
from ..t5.tokenization_t5 import T5Tokenizer
from ...configuration_utils import PreTrainedConfig
from ..vivit.configuration_vivit import VivitConfig
from ..vivit.modeling_vivit import (
    VivitEmbeddings,
    VivitAttention,
    VivitEncoder,
    VivitLayer,
    VivitPreTrainedModel,
    VivitTubeletEmbeddings,
)
from ..llava_onevision.video_processing_llava_onevision import LlavaOnevisionVideoProcessor
from ..siglip.configuration_siglip import SiglipConfig
from ..qwen3_next.modeling_qwen3_next import l2norm
# from ..siglip.modeling_siglip import lecun_normal


logger = logging.get_logger(__name__)


class VideoPrismVisionConfig(VivitConfig):
    model_type = "videoprism_vision_model"
    base_config_key = "vision_config"

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
        attn_logit_softcapping=50.0,
        num_auxiliary_layers=2,
        apply_l2_norm=True,
        num_labels=1000,
        **kwargs,
    ):
        super().__init__()
        self.num_spatial_layers = num_spatial_layers
        self.num_temporal_layers = num_temporal_layers
        self.attn_logit_softcapping = attn_logit_softcapping
        self.num_auxiliary_layers = num_auxiliary_layers
        self.apply_l2_norm = apply_l2_norm
        self.num_labels = num_labels
        del self.num_hidden_layers

class VideoPrismTextConfig(PreTrainedConfig):
    model_type = "videoprism_text_model"
    base_config_key = "text_config"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_unimodal_layers=12,
        vocabulary_size=32000,
        apply_l2_norm=True,
        hidden_act="relu",
        attention_probs_dropout_prob=0.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        layer_norm_eps=1e-06,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size=hidden_size
        self.intermediate_size=intermediate_size
        self.num_attention_heads=num_attention_heads
        self.num_unimodal_layers=num_unimodal_layers
        self.vocabulary_size=vocabulary_size
        self.apply_l2_norm=apply_l2_norm
        self.hidden_act=hidden_act
        self.attention_probs_dropout_prob=attention_probs_dropout_prob
        self.qkv_bias=qkv_bias
        self.hidden_dropout_prob=hidden_dropout_prob
        self.layer_norm_eps=layer_norm_eps
        self.initializer_range=initializer_range


class VideoPrismConfig(SiglipConfig):
    def __init__(self, text_config=None, vision_config=None, **kwargs):
        super().__init__(**kwargs)
        del self.initializer_factor
    

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



class VideoPrismVideoProcessor(LlavaOnevisionVideoProcessor):
    resample = PILImageResampling.BICUBIC  #! PILImageResampling.LANCZOS
    size = {"height": 288, "width": 288}
    do_normalize = False


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
        if not interpolate_pos_encoding and (height != self.image_size[0] or width != self.image_size[1]):
            raise ValueError(
                f"Image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}). Set interpolate_pos_encoding=True to automatically resize the model position embeddings."
            )
        # permute to (batch_size, num_channels, num_frames, height, width)
        pixel_values_videos = pixel_values_videos.permute(0, 2, 1, 3, 4)
        
        hidden_states = self.projection(pixel_values_videos)
        # flatten the spatial part and permute to (B, T, num_patches, dim) 
        hidden_states = hidden_states.flatten(3).permute(0, 2, 3, 1)
        # combine batch and time dimension
        batch_size, num_frames, num_patches, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(batch_size * num_frames, num_patches, hidden_size)
        
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
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.patch_embeddings.num_patches, config.hidden_size))

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

        num_row_patches = height // self.patch_size[0]
        num_col_patches = width // self.patch_size[1]

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = self.position_embeddings.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)        

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(num_row_patches, num_col_patches),
            mode="bilinear",
            antialias=True,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self, pixel_values_videos: torch.Tensor, interpolate_pos_encoding: bool = False):
        
        b, t, c, h, w = pixel_values_videos.shape
        assert h == w, "Input image height and width must be the same"
        embeddings = self.patch_embeddings(pixel_values_videos, interpolate_pos_encoding)
        
        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, h, w)
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

        self.position_embeddings = nn.Parameter(torch.zeros(1, self.config.num_frames, config.hidden_size))

    def interpolate_pos_encoding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Interpolates the embedding to the target sequence length
        """
        target_emb_length = embeddings.shape[1]
        source_emb_length = self.position_embeddings.shape[1]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and target_emb_length == source_emb_length:
            return self.position_embeddings

        source_emb = self.position_embeddings
        dim = embeddings.shape[-1]
        source_emb = source_emb.unsqueeze(1)
        source_emb = nn.functional.interpolate(
            source_emb,
            size=(target_emb_length, dim),
            mode="bilinear",
            antialias=True,
        )
        
        return source_emb.squeeze(1)


    def forward(self, pixel_values_videos: torch.Tensor, input_shape, interpolate_pos_encoding: bool = False):
        if input_shape is not None:
            b, t, c, h, w = input_shape

        _, features, dim = pixel_values_videos.shape

        hidden_states = pixel_values_videos.view(b, t, features, dim)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        embeddings = hidden_states.reshape(b * features, t, dim)
        
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
    softcap: Optional[float] = None,
    **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling                 

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask.expand(*attn_weights.shape)

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class VideoPrismSelfAttention(nn.Module):
    def __init__(self, config: VideoPrismConfig):
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
        self.scale = self.attention_head_size**-0.5
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        new_shape = batch_size, -1, self.num_attention_heads, self.attention_head_size
        query = self.query(hidden_states).view(*new_shape).transpose(1, 2)
        key = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        value = self.value(hidden_states).view(*new_shape).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        context_layer, attention_probs = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout_prob,
            softcap=self.config.attn_logit_softcapping,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer, attention_probs

class VideoPrismAttention(VivitAttention):

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self_attn_output, _ = self.attention(hidden_states, attention_mask)
        output = self.output(self_attn_output, hidden_states)
        return output


class VideoPrismLayerNorm(nn.LayerNorm):
    def forward(self, hidden_states: torch.Tensor):
        return F.layer_norm(
            hidden_states, self.normalized_shape, self.weight+1, self.bias, self.eps
        )


class VideoPrismLayer(VivitLayer):

    def __init__(self, config: VideoPrismVisionConfig):
        self.config = config
        super().__init__(config)
        del self.chunk_size_feed_forward
        del self.seq_len_dim
        self.layernorm_after = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_before = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states_norm = self.layernorm_before(hidden_states)
        attention_output = self.attention(hidden_states_norm, attention_mask)

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in VideoPrism, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        return layer_output

class VideoPrismSpatialEncoder(VivitEncoder):
    def __init__(self, config: VideoPrismVisionConfig):
        super().__init__(config)
        self.layer = nn.ModuleList([VideoPrismLayer(config) for _ in range(config.num_spatial_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> BaseModelOutput:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)

        return BaseModelOutput(last_hidden_state=hidden_states)


class VideoPrismTemporalEncoder(VivitEncoder):
    def __init__(self, config: VideoPrismVisionConfig):
        super().__init__(config)
        self.layer = nn.ModuleList([VideoPrismLayer(config) for _ in range(config.num_temporal_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> BaseModelOutput:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)

        return BaseModelOutput(last_hidden_state=hidden_states)

    
class VideoPrismAuxiliaryEncoder(VivitEncoder):
    def __init__(self, config: VideoPrismVisionConfig):
        super().__init__(config)
        self.layer = nn.ModuleList([VideoPrismLayer(config) for _ in range(config.num_auxiliary_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> BaseModelOutput:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)

        return BaseModelOutput(last_hidden_state=hidden_states)


class VideoPrismTextEncoder(VivitEncoder):
    def __init__(self, config: VideoPrismTextConfig):
        super().__init__(config)
        self.layer = nn.ModuleList([VideoPrismLayer(config) for _ in range(config.num_unimodal_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> BaseModelOutput:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)

        return BaseModelOutput(last_hidden_state=hidden_states)

@auto_docstring
class VideoPrismPreTrainedModel(VivitPreTrainedModel):
    base_model_prefix = "videoprism"
    main_input_name = "pixel_values_videos"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _supports_sdpa = True
    _supports_flash_attn = True
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
    def __init__(self, config: VideoPrismVisionConfig):
        super().__init__(config)
        self.config = config
        self.layernorm1 = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.spatial_embeddings = VideoPrismSpatialEmbeddings(config)
        self.temporal_embeddings = VideoPrismTemporalEmbeddings(config)
        self.spatial_encoder = VideoPrismSpatialEncoder(config)
        self.temporal_encoder = VideoPrismTemporalEncoder(config)
        self.post_init()

    @auto_docstring
    def forward(
        self,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> BaseModelOutputWithSpatialAndTemporalStates:
        if pixel_values_videos is None:
            raise ValueError("You have to specify pixel_values_videos")

        input_shape = pixel_values_videos.shape
        spatial_embeds = self.spatial_embeddings(pixel_values_videos, interpolate_pos_encoding)
        spatial_encoder_outputs: BaseModelOutput = self.spatial_encoder(hidden_states=spatial_embeds)
        spatial_sequence_output = spatial_encoder_outputs.last_hidden_state
        features = self.layernorm1(spatial_sequence_output)  # ? shape (B * T, 256, 768)

        temporal_embeds = self.temporal_embeddings(features, input_shape, interpolate_pos_encoding)
        temporal_encoder_outputs: BaseModelOutput = self.temporal_encoder(hidden_states=temporal_embeds)
        temporal_sequence_output = temporal_encoder_outputs.last_hidden_state
        features = self.layernorm2(temporal_sequence_output)
        _, num_frames, dim = features.shape
        features = features.view(input_shape[0], -1, num_frames, dim).permute(0, 2, 1, 3).contiguous()
        _, num_frames, num_patches, dim = features.shape
        features = features.view(input_shape[0], num_frames * num_patches, -1)

        return BaseModelOutputWithSpatialAndTemporalStates(
            last_hidden_state=features,
            temporal_hidden_state=temporal_sequence_output,
            spatial_hidden_state=spatial_sequence_output,
        )


class VideoPrismMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: VideoPrismConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.intermediate_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob = config.attention_probs_dropout_prob
        # PerDimScale
        self.dim = int(config.intermediate_size / config.num_attention_heads)
        self.per_dim_scale = nn.Parameter(torch.zeros(self.dim))
        r_softplus_0 = 1.442695041
        scale = torch.tensor(r_softplus_0 / (self.dim**0.5))
        softplus = nn.functional.softplus(self.per_dim_scale)
        scale = scale * softplus
        self.register_buffer("scale", scale)
        
        self.pooling_attention_query = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.query = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.qkv_bias)
        self.projection = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.qkv_bias)
        self.layernorm = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dim = int(config.intermediate_size / config.num_attention_heads)
        
    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        
        batch_size, seq_length, hidden_size = hidden_states.shape
        query = self.pooling_attention_query.expand(batch_size, -1, -1)
        query_layer = (
            self.query(query)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        query_layer = query_layer * self.scale.expand(*query_layer.shape)

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
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        
        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            scaling=1.0,
            dropout=0.0 if not self.training else self.dropout_prob,
            softcap=self.config.attn_logit_softcapping,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        outputs = self.projection(context_layer)
        outputs = self.layernorm(outputs)
        return (outputs, attention_probs)


class VideoPrismTextModel(VideoPrismPreTrainedModel):
    def __init__(self, config: VideoPrismTextConfig):
        super().__init__(config)
        self.config = config
        self.text_encoder = VideoPrismTextEncoder(self.config)
        self.token_embeddings = nn.Embedding(config.vocabulary_size, config.hidden_size)
        self.cls_emb = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.layernorm = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.normalize = config.apply_l2_norm
        self.post_init()

    def create_sinusoidal_positions(self, num_pos: int, dim: int) -> torch.Tensor:
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64) / (dim-2)))
        sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=torch.int64).float(), inv_freq).float()
        return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> BaseModelOutput:
        batch_size, seq_length = input_ids.shape
        hidden_states = self.token_embeddings(input_ids)
        hidden_states = hidden_states * (self.config.hidden_size**0.5)
        
        cls_padding = torch.ones(batch_size, 1)
        input_ids = torch.cat((input_ids, cls_padding), dim=1)
        attention_mask = torch.cat((attention_mask, cls_padding), dim=1) if attention_mask is not None else None

        if attention_mask is not None:
            attention_mask = create_causal_mask(
                                config=self.config,
                                input_embeds=hidden_states,
                                attention_mask=attention_mask,
                                cache_position=torch.arange(hidden_states.shape[1], device=hidden_states.device),
                                past_key_values=None,
                            )

        features = hidden_states + self.create_sinusoidal_positions(seq_length, self.config.hidden_size)
        cls_emb = self.cls_emb * (self.config.hidden_size**0.5)
        cls_emb = cls_emb.expand(features.shape[0], -1, -1)
        features = torch.cat((features, cls_emb), dim=1)
        text_encoder_output = self.text_encoder(features, attention_mask)
        features = text_encoder_output.last_hidden_state
        features = self.layernorm(features)
        text_embeddings = features[:, -1]
        
        if self.normalize:
            text_embeddings = l2norm(text_embeddings, dim=-1)

        return BaseModelOutput(
            last_hidden_state=text_embeddings,
        )


class VideoPrismVideoModel(VideoPrismPreTrainedModel):
    def __init__(self, config: VideoPrismVisionConfig):
        super().__init__(config)
        self.config = config
        self.backbone = VideoPrismModel(config)
        self.auxiliary_encoder = VideoPrismAuxiliaryEncoder(config)
        self.contrastive_vision_pooler = VideoPrismMultiheadAttentionPoolingHead(config)
        self.normalize = config.apply_l2_norm
        self.post_init()

    def forward(
        self,
        pixel_values_videos: torch.FloatTensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ) -> BaseModelOutput:

        backbone_outputs = self.backbone(pixel_values_videos=pixel_values_videos)      
        video_features = backbone_outputs.last_hidden_state
        auxiliary_output = self.auxiliary_encoder(video_features)
        auxiliary_output_features = auxiliary_output.last_hidden_state
        contrastive_vision_pooler_output = self.contrastive_vision_pooler(auxiliary_output_features)
        video_embeddings = contrastive_vision_pooler_output[0]
        if self.normalize:
            video_embeddings = l2norm(video_embeddings, dim=-1)

        return VideoPrismVideoOutput(
            video_last_hidden_state=video_embeddings,
            auxiliary_output=auxiliary_output,
            attention_pooling_output=contrastive_vision_pooler_output,
            )


class VideoPrismClipModel(VideoPrismPreTrainedModel):   
    def __init__(self, config: VideoPrismConfig):
        super().__init__(config)
        self.config = config
        self.video_model = VideoPrismVideoModel(config.vision_config)
        self.text_model = VideoPrismTextModel(config.text_config)
        self.post_init()

    def forward(
        self,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ) -> VideoPrismClipOutput:

        video_model_outputs = self.video_model(pixel_values_videos=pixel_values_videos)
        text_model_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)

        video_embeddings = video_model_outputs.video_last_hidden_state
        text_embeddings = text_model_outputs.last_hidden_state
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


class VideoPrismForVideoClassification(VideoPrismPreTrainedModel):
    def __init__(self, config: VideoPrismConfig):
        super().__init__(config)
        self.encoder = VideoPrismModel(config)
        self.contrastive_vision_pooler = VideoPrismMultiheadAttentionPoolingHead(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput:
        encoder_outputs = self.encoder(pixel_values_videos=pixel_values_videos)
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.contrastive_vision_pooler(sequence_output).pooled_output
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config, **kwargs)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.last_hidden_state,
        )


__all__ = [
    "VideoPrismVisionConfig",
    "VideoPrismTextConfig",
    "VideoPrismConfig",
    "VideoPrismModel",
    "VideoPrismPreTrainedModel",
    "VideoPrismClipModel",
    "VideoPrismForVideoClassification",
    "VideoPrismTokenizer",
    "VideoPrismVideoProcessor",
]