# coding=utf-8
# Copyright 2026 Microsoft Research and The HuggingFace Inc. team.
# Licensed under the MIT License.

"""BEATs model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging

logger = logging.get_logger(__name__)


class BEATsConfig(PretrainedConfig):
    """
    Configuration class for BEATs model.

    Args:
        input_patch_size (`int`, *optional*, defaults to 16):
            Patch size for the patch embedding.
        embed_dim (`int`, *optional*, defaults to 512):
            Dimension of the patch embedding.
        conv_bias (`bool`, *optional*, defaults to `False`):
            Whether to include bias in conv encoder.
        encoder_layers (`int`, *optional*, defaults to 12):
            Number of transformer encoder layers.
        encoder_embed_dim (`int`, *optional*, defaults to 768):
            Encoder embedding dimension.
        encoder_ffn_embed_dim (`int`, *optional*, defaults to 3072):
            Encoder FFN embedding dimension.
        encoder_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads.
        activation_fn (`str`, *optional*, defaults to `"gelu"`):
            Activation function.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            Attention dropout probability.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            Activation dropout probability.
        dropout_input (`float`, *optional*, defaults to 0.1):
            Dropout applied to the input features.
        encoder_layerdrop (`float`, *optional*, defaults to 0.05):
            LayerDrop probability for encoder layers.
        layer_norm_first (`bool`, *optional*, defaults to `False`):
            Whether to apply layer norm first.
        deep_norm (`bool`, *optional*, defaults to `True`):
            Whether to apply deep norm.
        relative_position_embedding (`bool`, *optional*, defaults to `True`):
            Whether to use relative position embeddings.
        num_buckets (`int`, *optional*, defaults to 320):
            Number of buckets for relative position embedding.
        max_distance (`int`, *optional*, defaults to 800):
            Maximum distance for relative position embedding.
        gru_rel_pos (`bool`, *optional*, defaults to `True`):
            Whether to use GRU-based relative position gating.
        conv_pos (`int`, *optional*, defaults to 128):
            Number of filters for convolutional positional embeddings.
        conv_pos_groups (`int`, *optional*, defaults to 16):
            Number of groups for convolutional positional embeddings.
        num_classes (`int`, *optional*, defaults to 527):
            Number of output classes for classification.
    """

    model_type = "beats"

    def __init__(
        self,
        input_patch_size: int = 16,
        embed_dim: int = 512,
        conv_bias: bool = False,
        encoder_layers: int = 12,
        encoder_embed_dim: int = 768,
        encoder_ffn_embed_dim: int = 3072,
        encoder_attention_heads: int = 12,
        activation_fn: str = "gelu",
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        dropout_input: float = 0.1,
        encoder_layerdrop: float = 0.05,
        layer_norm_first: bool = False,
        deep_norm: bool = True,
        relative_position_embedding: bool = True,
        num_buckets: int = 320,
        max_distance: int = 800,
        gru_rel_pos: bool = True,
        grep_linear_units: int = 8,
        conv_pos: int = 128,
        conv_pos_groups: int = 16,
        num_classes: int = 527,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_patch_size = input_patch_size
        self.embed_dim = embed_dim
        self.conv_bias = conv_bias
        self.encoder_layers = encoder_layers
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout_input = dropout_input
        self.encoder_layerdrop = encoder_layerdrop
        self.layer_norm_first = layer_norm_first
        self.deep_norm = deep_norm
        self.relative_position_embedding = relative_position_embedding
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.gru_rel_pos = gru_rel_pos
        self.grep_linear_units = grep_linear_units
        self.conv_pos = conv_pos
        self.conv_pos_groups = conv_pos_groups
        self.num_classes = num_classes

__all__ = ['BEATsConfig']
