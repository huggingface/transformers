# coding=utf-8
# Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.
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

import copy
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch import nn

from transformers.models.blip.image_processing_blip import BlipImageProcessor

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...generation import ClassifierFreeGuidanceLogitsProcessor, GenerationMixin, GenerationMode, LogitsProcessorList
from ...generation.utils import GenerateDecoderOnlyOutput
from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
)
from ...modeling_outputs import ModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torch_available,
    is_vision_available,
    logging,
)
from ..auto import AutoModel
from ..blip_2.modeling_blip_2 import Blip2VisionModel
from ..chameleon.configuration_chameleon import ChameleonVQVAEConfig
from ..chameleon.modeling_chameleon import (
    ChameleonVQVAE,
    ChameleonVQVAEEncoderAttnBlock,
    ChameleonVQVAEEncoderConvDownsample,
    ChameleonVQVAEEncoderResnetBlock,
    ChameleonVQVAEVectorQuantizer,
)
from ..idefics.modeling_idefics import IdeficsBaseModelOutputWithPast, IdeficsCausalLMOutputWithPast
from ..llama.modeling_llama import eager_attention_forward
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..siglip.modeling_siglip import SiglipEncoder, SiglipEncoderLayer, SiglipVisionEmbeddings


if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.checkpoint

if is_vision_available():
    import PIL

from ...configuration_utils import PretrainedConfig
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)

# General docstring


class JanusVisionConfig(SiglipVisionConfig):
    r"""
    This is the configuration class to store the configuration of a [`JanusVisionModel`]. It is used to instantiate a
    `JanusVisionModel` according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        image_size (`int`, *optional*, defaults to 384):
            The size (resolution) of each image.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"`, and `"gelu_new"` are supported.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of MLP hidden dimensionality to embedding dimensionality.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys, and values in the attention layers.
        hidden_dropout_rate (`float`, *optional*, defaults to 0.0):
            The dropout probability for fully connected layers in the encoder.
        projection_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the MLP projection head.
        projection_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for the projection layer.
        use_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to normalize the query and key matrices.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer for initializing all weight matrices.
        depth (`int`, *optional*, defaults to 2):
            Number of hidden layers in the aligner module.
        num_image_tokens (`int`, *optional*, defaults to 576):
            Number of image tokens.
    """

    model_type = "janus_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        patch_size=16,
        image_size=384,
        attention_dropout=0.0,
        layer_norm_eps=1e-6,
        hidden_act="gelu",
        mlp_ratio=4.0,
        attention_bias=True,
        hidden_dropout_rate=0.0,
        projection_dim=2048,
        projection_dropout=0.0,
        use_qk_norm=False,
        initializer_range=0.02,
        depth=2,
        num_image_tokens=576,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_channels=num_channels,
            patch_size=patch_size,
            image_size=image_size,
            attention_dropout=attention_dropout,
            layer_norm_eps=layer_norm_eps,
            hidden_act=hidden_act,
            **kwargs,
        )
        del self.intermediate_size

        self.mlp_ratio = mlp_ratio
        self.attention_bias = attention_bias
        self.hidden_dropout_rate = hidden_dropout_rate
        self.projection_dim = projection_dim
        self.projection_dropout = projection_dropout
        self.use_qk_norm = use_qk_norm
        self.initializer_range = initializer_range
        self.depth = depth
        self.num_image_tokens = num_image_tokens


class JanusVQVAEConfig(ChameleonVQVAEConfig):
    r"""
    This is the configuration class to store the configuration of a [`JanusVQVAEModel`]. It is used to instantiate a
    `JanusVQVAEModel` according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information. Instantiating a
    configuration with the defaults will yield a similar configuration to the VQModel of the
    [deepseek-community/Janus-Pro-1B](https://huggingface.co/deepseek-community/Janus-Pro-1B).

    Args:
        embed_dim (`int`, *optional*, defaults to 8):
            Dimensionality of each embedding vector.
        num_embeddings (`int`, *optional*, defaults to 16384):
            Number of codebook embeddings.
        double_latent (`bool`, *optional*, defaults to `False`):
            Whether to use double z channels.
        latent_channels (`int`, *optional*, defaults to 256):
            Number of channels for the latent space.
        num_patches (`int`, *optional*, defaults to 32):
            Num of patches the input images can be divided into.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            Number of out channels.
        base_channels (`int`, *optional*, defaults to 128):
            Base channel count.
        channel_multiplier (`list[int]`, *optional*, defaults to `[1, 1, 2, 2, 4]`):
            Channel multipliers for each resolution.
        num_res_blocks (`int`, *optional*, defaults to 2):
            Number of residual blocks.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        projection_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the MLP projection head.
        num_hidden_layers (`int`, *optional*, defaults to 2):
            Number of hidden layers in VAVAE MLP Connecter module.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        image_token_embed_dim (`int`, *optional*, defaults to 2048):
            Dimension of image embeddings. It should be same as the dimensionality of text embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 8,
        num_embeddings: int = 16384,
        double_latent: bool = False,
        latent_channels: int = 256,
        num_patches: int = 32,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multiplier: list[int] = [1, 1, 2, 2, 4],
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        initializer_range=0.02,
        projection_dim=2048,
        num_hidden_layers=2,
        hidden_act="gelu",
        image_token_embed_dim=2048,
        **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_embeddings=num_embeddings,
            double_latent=double_latent,
            latent_channels=latent_channels,
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multiplier=channel_multiplier,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            initializer_range=initializer_range,
            **kwargs,
        )
        self.num_patches = num_patches
        self.out_channels = out_channels
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.image_token_embed_dim = image_token_embed_dim

        del self.resolution
        del self.attn_resolutions
        del self.attn_type


class JanusConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`JanusModel`]. It is used to instantiate an
    Janus model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Janus-1B or Janus-7B models.

    e.g. [deepseek-community/Janus-Pro-1B](https://huggingface.co/deepseek-community/Janus-Pro-1B) or
    [deepseek-community/Janus-Pro-7B](https://huggingface.co/deepseek-community/Janus-Pro-7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `JanusVisionConfig`):
            The config object or dictionary of the vision backbone.
        vq_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `JanusVQVAEConfig`):
            The config object or dictionary of the VQVAE backbone.
        image_token_id (`int`, *optional*, defaults to 100581):
            Token index of a placeholder image token.

    Example:

    ```python
    >>> from transformers import JanusForConditionalGeneration, JanusConfig, JanusVisionConfig, JanusVQVAEConfig, LlamaConfig

    >>> # Initializing a Janus vision config
    >>> vision_config = JanusVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a VQ config
    >>> vq_config = JanusVQVAEConfig()

    >>> # Initializing a Janus Pro 1B style configuration
    >>> configuration = JanusConfig(vision_config=vision_config, text_config=text_config, vq_config=vq_config)

    >>> # Initializing a model from the Janus Pro 1B style configuration
    >>> model = JanusForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "janus"
    sub_configs = {
        "text_config": AutoConfig,
        "vision_config": JanusVisionConfig,
        "vq_config": JanusVQVAEConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        vq_config=None,
        image_token_id=100581,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)

        elif text_config is None:
            logger.info("`text_config` is None. Initializing with default values")
            self.text_config = CONFIG_MAPPING["llama"]()
        elif isinstance(text_config, PretrainedConfig):
            self.text_config = text_config
        else:
            raise ValueError(
                f"Invalid type for `text_config`. Must be either `dict` or `LlamaConfig`."
                f" Type found: {type(text_config)}"
            )

        if vision_config is None:
            logger.info("`vision_config` is None. Initializing with default JanusVisionConfig values")
            self.vision_config = JanusVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = JanusVisionConfig(**vision_config)
        elif isinstance(vision_config, JanusVisionConfig):
            self.vision_config = vision_config
        else:
            raise ValueError(
                f"Invalid type for `vision_config`. Must be either `dict` or `JanusVisionConfig`."
                f" Type found: {type(vision_config)}"
            )

        if vq_config is None:
            logger.info("`vq_config` is None. Initializing with default JanusVQVAEConfig values")
            self.vq_config = JanusVQVAEConfig()
        elif isinstance(vq_config, dict):
            self.vq_config = JanusVQVAEConfig(**vq_config)
        elif isinstance(vq_config, JanusVQVAEConfig):
            self.vq_config = vq_config
        else:
            raise ValueError(
                f"Invalid type for `vq_config`. Must be either `dict` or `JanusVQVAEConfig`."
                f" Type found: {type(vq_config)}"
            )

        self.initializer_range = self.vision_config.initializer_range
        # This dimension is required when decoding discrete image tokens to continuous input.
        self.vq_config.num_patches = self.vision_config.image_size // self.vision_config.patch_size
        # The default is only the index for the 1B model, 7B uses a different one
        self.image_token_id = image_token_id
        super().__init__(**kwargs)


@auto_docstring
class JanusPreTrainedModel(PreTrainedModel):
    config: JanusConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer", "JanusVisionEncoderLayer"]
    _skip_keys_device_placement = ["past_key_values", "causal_mask"]
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_param_buffer_assignment = False


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Janus VQ-VAE mode model outputs.
    """
)
class JanusVQVAEOutput(ModelOutput):
    r"""
    decoded_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
        Reconstructed pixel values after encoding and decoding the input.
    embedding_loss (`torch.FloatTensor`):
        Embedding loss.
    """

    decoded_pixel_values: Optional[torch.FloatTensor] = None
    embedding_loss: torch.FloatTensor = None


class JanusBaseModelOutputWithPast(IdeficsBaseModelOutputWithPast):
    pass


class JanusCausalLMOutputWithPast(IdeficsCausalLMOutputWithPast):
    pass


class JanusVisionEmbeddings(SiglipVisionEmbeddings):
    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        if interpolate_pos_encoding:
            pos_embeds = self.interpolate_pos_encoding(embeddings, height, width)
        else:
            pos_embeds = self.position_embedding(self.position_ids)

        embeddings = embeddings + pos_embeds

        return embeddings


class JanusVisionAttention(nn.Module):
    """Attention Class for Janus Vision Encoder"""

    def __init__(self, config: JanusVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        proj_dropout = config.projection_dropout
        qk_norm = config.use_qk_norm
        self.is_causal = False

        # Janus has no MHA, hence for `eager_attention_forward` call setting `num_key_value_groups` to 1.
        self.num_key_value_groups = 1

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.projection_layer = nn.Linear(self.embed_dim, self.embed_dim)
        self.projection_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

        self.q_norm = nn.LayerNorm(self.embed_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.embed_dim) if qk_norm else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(-1, self.num_heads, self.head_dim)
        query_states = self.q_norm(query_states)

        key_states = key_states.reshape(-1, self.num_heads, self.head_dim)
        key_states = self.k_norm(key_states)

        query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scale,
            is_causal=self.is_causal,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        output = self.projection_layer(attn_output)
        output = self.projection_dropout(output)
        return output, attn_weights


class JanusVisionMLP(nn.Module):
    def __init__(self, config: JanusVisionConfig):
        super().__init__()
        self.config = config
        self.intermediate_size = int(config.hidden_size * config.mlp_ratio)
        self.activation_fn = ACT2FN[config.hidden_act]  # Gelu act
        self.fc1 = nn.Linear(config.hidden_size, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_rate)
        self.dropout2 = nn.Dropout(config.hidden_dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.dropout1(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        return hidden_states


class JanusVisionEncoderLayer(SiglipEncoderLayer):
    def __init__(self, config: JanusVisionConfig):
        super().__init__(config)
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = JanusVisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = JanusVisionMLP(config)


class JanusVisionEncoder(SiglipEncoder):
    def __init__(self, config: JanusVisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([JanusVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])


class JanusVisionModel(Blip2VisionModel):
    def __init__(self, config: JanusVisionConfig):
        super().__init__(config)
        self.encoder = JanusVisionEncoder(config)


class JanusVisionAlignerMLP(nn.Module):
    def __init__(self, config: JanusVisionConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.hidden_size, config.projection_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(config.projection_dim, config.projection_dim) for _ in range(1, config.depth)]
        )
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        for layer in self.hidden_layers:
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = layer(hidden_states)
        return hidden_states


class JanusVQVAEVectorQuantizer(ChameleonVQVAEVectorQuantizer):
    def __init__(self, config: JanusVQVAEConfig):
        super().__init__(config)
        self.quant_state_dims = [config.num_patches] * 2

    def get_codebook_entry(self, image_tokens: torch.LongTensor) -> torch.FloatTensor:
        batch_size = image_tokens.shape[0]
        emb_dim: int = self.embedding.weight.shape[-1]

        # get quantized latent vectors
        hidden_state_quant = self.embedding(image_tokens)
        # l2 normalization on the last dimension
        hidden_state_quant = F.normalize(hidden_state_quant, p=2, dim=-1)

        # reshape back to match original input shape
        hidden_state_quant = hidden_state_quant.view((batch_size, *self.quant_state_dims, emb_dim))
        hidden_state_quant = hidden_state_quant.permute(0, 3, 1, 2).contiguous()

        return hidden_state_quant


class JanusVQVAEResnetBlock(ChameleonVQVAEEncoderResnetBlock):
    pass


class JanusVQVAEAttnBlock(ChameleonVQVAEEncoderAttnBlock):
    pass


class JanusVQVAEConvDownsample(ChameleonVQVAEEncoderConvDownsample):
    pass


class JanusVQVAEConvUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states):
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)
        return hidden_states


class JanusVQVAEMidBlock(nn.Module):
    def __init__(self, config: JanusVQVAEConfig, channels: int):
        super().__init__()
        self.block_1 = JanusVQVAEResnetBlock(
            config=config,
            in_channels=channels,
            out_channels=channels,
        )
        self.attn_1 = JanusVQVAEAttnBlock(channels)
        self.block_2 = JanusVQVAEResnetBlock(
            config=config,
            in_channels=channels,
            out_channels=channels,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.block_1(hidden_states)
        hidden_states = self.attn_1(hidden_states)
        hidden_states = self.block_2(hidden_states)
        return hidden_states


class JanusVQVAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        base_channels = config.base_channels
        in_channels = config.in_channels
        double_latent = config.double_latent
        latent_channels = config.latent_channels
        channel_multiplier = config.channel_multiplier

        self.conv_in = torch.nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        in_channel_multiplier = (1,) + tuple(channel_multiplier)
        self.in_channel_multiplier = in_channel_multiplier
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = base_channels * in_channel_multiplier[i_level]
            block_out = base_channels * channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    JanusVQVAEResnetBlock(
                        config=config,
                        in_channels=block_in,
                        out_channels=block_out,
                    )
                )
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn.append(JanusVQVAEAttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = JanusVQVAEConvDownsample(block_in)
            self.down.append(down)

        self.mid = JanusVQVAEMidBlock(config, block_in)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * latent_channels if double_latent else latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, pixel_values: torch.LongTensor):
        # downsampling
        hidden_states = [self.conv_in(pixel_values)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                hidden_state = self.down[i_level].block[i_block](
                    hidden_states[-1],
                )
                if len(self.down[i_level].attn) > 0:
                    hidden_state = self.down[i_level].attn[i_block](hidden_state)
                hidden_states.append(hidden_state)
            if i_level != self.num_resolutions - 1:
                hidden_states.append(self.down[i_level].downsample(hidden_states[-1]))

        # middle
        last_hidden_state = hidden_states[-1]
        last_hidden_state = self.mid(last_hidden_state)

        # end
        last_hidden_state = self.norm_out(last_hidden_state)
        last_hidden_state *= torch.sigmoid(last_hidden_state)
        last_hidden_state = self.conv_out(last_hidden_state)
        return last_hidden_state


class JanusVQVAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        base_channels = config.base_channels
        latent_channels = config.latent_channels
        out_channels = config.out_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = base_channels * config.channel_multiplier[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = torch.nn.Conv2d(latent_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = JanusVQVAEMidBlock(config, block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = base_channels * config.channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    JanusVQVAEResnetBlock(
                        config=config,
                        in_channels=block_in,
                        out_channels=block_out,
                    )
                )
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn.append(JanusVQVAEAttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = JanusVQVAEConvUpsample(block_in)
            self.up.append(up)

        # end
        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = torch.nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_state: torch.FloatTensor) -> torch.FloatTensor:
        hidden_state = self.conv_in(hidden_state)

        # middle
        hidden_state = self.mid(hidden_state)

        # upsampling
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks + 1):
                hidden_state = self.up[i_level].block[i_block](hidden_state)
                if len(self.up[i_level].attn) > 0:
                    hidden_state = self.up[i_level].attn[i_block](hidden_state)
            if i_level != self.num_resolutions - 1:
                hidden_state = self.up[i_level].upsample(hidden_state)

        hidden_state = self.norm_out(hidden_state)
        hidden_state *= torch.sigmoid(hidden_state)
        hidden_state = self.conv_out(hidden_state)
        return hidden_state


class JanusVQVAE(ChameleonVQVAE):
    _no_split_modules = [
        "JanusVQVAEAttnBlock",
        "JanusVQVAEResnetBlock",
        "JanusVQVAEVectorQuantizer",
    ]
    main_input_name = "pixel_values"

    def __init__(self, config: JanusVQVAEConfig):
        super().__init__(config)
        self.decoder = JanusVQVAEDecoder(config)
        self.gradient_checkpointing = False

        # Initialize the VQVAE model.
        self.post_init()

    def decode(self, image_tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Decodes quantized token IDs into pixel values.
        Args:
            image_tokens (torch.LongTensor): Batch of token IDs.
        Returns:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                Pixel values decoded from the token IDs.
        """
        if image_tokens.shape[1] != self.quantize.quant_state_dims[0] * self.quantize.quant_state_dims[1]:
            raise ValueError(
                f"Expected `image_tokens` to have shape `(batch_size, {self.quantize.quant_state_dims[0] * self.quantize.quant_state_dims[1]})`, "
                f"but got shape `{image_tokens.shape}`."
            )
        codebook_entry = self.quantize.get_codebook_entry(image_tokens)
        hidden_states = self.post_quant_conv(codebook_entry)
        pixel_values = self.decoder(hidden_states)
        return pixel_values

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        batch_size = pixel_values.shape[0]
        quant, embedding_loss, indices = self.encode(pixel_values)
        decoded_pixel_values = self.decode(indices.view(batch_size, -1))

        return JanusVQVAEOutput(decoded_pixel_values, embedding_loss)


class JanusVQVAEAlignerMLP(nn.Module):
    def __init__(self, config: JanusVQVAEConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.embed_dim, config.projection_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(config.projection_dim, config.projection_dim) for _ in range(1, config.num_hidden_layers)]
        )
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        for layer in self.hidden_layers:
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = layer(hidden_states)
        return hidden_states


class JanusVQVAEHead(nn.Module):
    """Head used for sampling tokens in image generation, replacing the usual lm head."""

    def __init__(self, config: JanusVQVAEConfig):
        super().__init__()
        self.proj_out = nn.Linear(config.image_token_embed_dim, config.projection_dim)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.vision_head = nn.Linear(config.projection_dim, config.num_embeddings)

    def forward(self, hidden_states: torch.Tensor) -> torch.tensor:
        hidden_states = self.proj_out(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.vision_head(hidden_states)
        return hidden_states


@auto_docstring(
    custom_intro="""
    The Janus model which consists of a siglip vision backbone, a Llama language model and a VQ model.
    """
)
class JanusModel(JanusPreTrainedModel):
    def __init__(self, config: JanusConfig):
        super().__init__(config)
        self.config = config
        # This is necessary for backward compatibility, see SiglipModel initialization
        self.vision_model = JanusVisionModel._from_config(config.vision_config)
        self.aligner = JanusVisionAlignerMLP(self.vision_model.config)

        self.vqmodel = JanusVQVAE._from_config(config.vq_config)

        # Below generation_* modules are used for Image generation.
        # Embeddings used for image generation, instead of Janus vision embeddings.
        self.generation_embeddings = nn.Embedding(self.vqmodel.config.num_embeddings, self.vqmodel.config.embed_dim)
        self.generation_aligner = JanusVQVAEAlignerMLP(self.vqmodel.config)
        self.generation_head = JanusVQVAEHead(self.vqmodel.config)

        self.language_model = AutoModel.from_config(config=config.text_config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing.
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(self, pixel_values):
        image_embeds = self.vision_model(pixel_values)
        image_embeds = self.aligner(image_embeds.last_hidden_state)
        return image_embeds

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            n_image_features = image_features.shape[0] * image_features.shape[1]
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values)
            image_features = image_embeds.reshape(-1, inputs_embeds.shape[-1])
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_attention_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_attention_mask, image_features)

        lm_output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        return JanusBaseModelOutputWithPast(
            last_hidden_state=lm_output.last_hidden_state,
            past_key_values=lm_output.past_key_values,
            hidden_states=lm_output.hidden_states,
            attentions=lm_output.attentions,
            image_hidden_states=image_embeds if pixel_values is not None else None,
        )


class JanusForConditionalGeneration(JanusPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["model.language_model.embed_tokens.weight", "lm_head.weight"]
    _can_compile_fullgraph = True

    def __init__(self, config: JanusConfig):
        super().__init__(config)
        self.config = config
        self.model = JanusModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # Initialize weights and apply final processing.
        self.post_init()

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def prepare_embeddings_for_image_generation(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_state = self.model.generation_embeddings(inputs)
        hidden_state = self.model.generation_aligner(hidden_state)
        return hidden_state

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return JanusCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        pixel_values=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- extra custom processing

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values

        return model_inputs

    def decode_image_tokens(self, image_tokens: torch.Tensor):
        """
        Decodes generated image tokens from language model to continuous pixel values
        with VQGAN module via upsampling.
        Args:
            image_tokens (`torch.LongTensor` of shape `(batch_size, num_of_tokens)`):
                The tensors corresponding to the input images.
        """
        decoded_image = self.model.vqmodel.decode(image_tokens)
        decoded_image = decoded_image.permute(0, 2, 3, 1)
        return decoded_image

    @torch.no_grad
    def generate(
        self,
        inputs: torch.Tensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        **kwargs,
    ):
        # 1. Handle generation config and model kwargs
        generation_config = kwargs.pop("generation_config", self.generation_config)
        generation_config = copy.deepcopy(generation_config)

        # Default to "text" generation if mode isn't provided
        generation_mode = kwargs.pop("generation_mode", "text")
        if generation_mode == "text":
            # Set guidance_scale=None to prevent running UnbatchedCFG processor.
            return super().generate(
                inputs=inputs,
                attention_mask=attention_mask,
                generation_config=generation_config,
                guidance_scale=None,
                **kwargs,
            )

        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs

        # Validate generation mode
        if generation_config.get_generation_mode() not in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            raise ValueError(
                "Got incompatible mode for Image Generation, should be one of greedy or sampling. "
                "Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`."
            )

        # Validate the configuration and model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Initialize logit processors
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()

        # Set `use_cache=True` as we will be using input embeds for generation.
        model_kwargs["use_cache"] = True

        if generation_config.guidance_scale is None:
            logger.warning("`guidance_scale` is required for CFG but not provided. Setting to default value of 5.")
            generation_config.guidance_scale = 5
        model_kwargs["guidance_scale"] = generation_config.guidance_scale

        # 3. Prepare model inputs
        input_ids, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        dtype, device = input_ids.dtype, input_ids.device

        if len(input_ids.shape) != 2:
            raise ValueError(
                f"Expected input ids of shape (batch_size, seq_len), but got {input_ids.shape}"
                "Passing `inputs embeds` is not supported currently."
            )

        # Prepare special tokens which will be used generate internally.
        kwargs_has_attention_mask = attention_mask is not None
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=input_ids.device)

        # 4. Add CFG processor along with user passed logit processor.
        if generation_config.guidance_scale and generation_config.guidance_scale > 1:
            logits_processor.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
            generation_config.guidance_scale = None  # Reset to prevent processor duplication.

        # 5. Prepare logits processor
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids.shape[1],
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
            device=device,
        )

        # 6. Expand inputs for multiple image generations per prompt.
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            expand_size=generation_config.num_return_sequences,
            **model_kwargs,
        )

        # 7. Prepare input and model caches
        num_image_tokens = self.model.vision_model.config.num_image_tokens
        batch_size, seq_len = input_ids.shape

        input_tokens = input_ids.repeat(2, 1)  # Double batch size for conditional/unconditional logits
        attention_mask = model_kwargs.pop("attention_mask", None)
        attention_mask = attention_mask.repeat(2, 1)
        model_kwargs["attention_mask"] = attention_mask

        # Mask all the tokens that are neither BOS nor BOI with pad token in the unconditional logits.
        mask = (input_tokens[batch_size:, :] != generation_config.bos_token_id) & (
            input_tokens[batch_size:, :] != generation_config.generation_kwargs["boi_token_id"]
        )
        input_tokens[batch_size:, :].masked_fill_(mask, generation_config.pad_token_id)

        inputs_embeds = self.get_input_embeddings()(input_tokens)

        model_kwargs = self._get_initial_cache_position(seq_len, device, model_kwargs)

        if model_kwargs.get("past_key_values", None) is None:
            # Prepare cache if not provided.
            model_kwargs["past_key_values"] = self._get_cache(
                cache_implementation=generation_config.cache_implementation or "static",
                # batch_size should account for both conditional/unconditional input; hence multiplied by 2.
                batch_size=batch_size * 2,
                # we should have at least a cache len of seq_len + num_image_tokens.
                max_cache_len=max(generation_config.max_length, num_image_tokens + seq_len),
                model_kwargs=model_kwargs,
            )

        # Placeholder for generated tokens.
        generated_tokens = torch.zeros((batch_size, num_image_tokens), dtype=dtype, device=device)

        # 8. init attention / hidden states / scores tuples
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        raw_scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None

        for i in range(num_image_tokens):
            model_inputs = self.prepare_inputs_for_generation(
                inputs_embeds=inputs_embeds, input_ids=input_tokens, **model_kwargs
            )

            model_inputs["attention_mask"] = model_inputs["attention_mask"].to(inputs_embeds.device)
            model_inputs["cache_position"] = model_inputs["cache_position"].to(inputs_embeds.device)

            outputs = self.model.language_model(
                **model_inputs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            # Update model_kwargs like cache_position for next generation.
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
            hidden_state = outputs.last_hidden_state[:, -1, :].clone()

            # Generate scores using the generation head (Not using above defined LM Head)
            scores = self.model.generation_head(hidden_state)
            next_token_scores = logits_processor(input_ids, scores)

            # Sample next token.
            if generation_config.do_sample:
                probs = torch.softmax(next_token_scores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(next_token_scores, dim=-1)

            generated_tokens[:, i] = next_token

            # Prepare embeddings for the next step.
            next_token = torch.cat([next_token, next_token])
            next_token = next_token.unsqueeze(-1)

            inputs_embeds = self.prepare_embeddings_for_image_generation(next_token)

        if return_dict_in_generate:
            if output_scores:
                raw_scores += (scores,)
            if output_logits:
                raw_logits += (hidden_state.float(),)
            if output_attentions:
                decoder_attentions += outputs.attentions
            if output_hidden_states:
                decoder_hidden_states += outputs.hidden_states

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=generated_tokens,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=outputs.past_key_values,
            )
        else:
            return generated_tokens


class JanusImageProcessor(BlipImageProcessor):
    r"""
    Constructs a JANUS image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 384, "width": 384}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        min_size (`int`, *optional*, defaults to 14):
            The minimum allowed size for the resized image. Ensures that neither the height nor width
            falls below this value after resizing.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        min_size: int = 14,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.min_size = min_size
        if image_mean is None:
            self.background_color = (127, 127, 127)
        else:
            self.background_color = tuple(int(x * 255) for x in image_mean)

    def pad_to_square(
        self,
        image: np.ndarray,
        background_color: Union[int, tuple[int, int, int]] = 0,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.array:
        """
        Pads an image to a square based on the longest edge.

        Args:
            image (`np.ndarray`):
                The image to pad.
            background_color (`int` or `tuple[int, int, int]`, *optional*, defaults to 0):
                The color to use for the padding. Can be an integer for single channel or a
                tuple of integers representing for multi-channel images. If passed as integer
                in mutli-channel mode, it will default to `0` in subsequent channels.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The padded image.
        """
        height, width = get_image_size(image, input_data_format)
        num_channels = image.shape[0] if input_data_format == ChannelDimension.FIRST else image.shape[-1]

        if height == width:
            image = (
                to_channel_dimension_format(image, data_format, input_data_format)
                if data_format is not None
                else image
            )
            return image

        max_dim = max(height, width)

        # Ensure background_color is the correct shape
        if isinstance(background_color, int):
            background_color = [background_color]
        elif len(background_color) != num_channels:
            raise ValueError(
                f"background_color must have no more than {num_channels} elements to match the number of channels"
            )

        if input_data_format == ChannelDimension.FIRST:
            result = np.zeros((num_channels, max_dim, max_dim), dtype=image.dtype)
            for i, color in enumerate(background_color):
                result[i, :, :] = color
            if width > height:
                start = (max_dim - height) // 2
                result[:, start : start + height, :] = image
            else:
                start = (max_dim - width) // 2
                result[:, :, start : start + width] = image
        else:
            result = np.zeros((max_dim, max_dim, num_channels), dtype=image.dtype)
            for i, color in enumerate(background_color):
                result[:, :, i] = color
            if width > height:
                start = (max_dim - height) // 2
                result[start : start + height, :, :] = image
            else:
                start = (max_dim - width) // 2
                result[:, start : start + width, :] = image

        return result

    def resize(
        self,
        image: np.ndarray,
        size: Union[dict[str, int], int],
        background_color: Optional[tuple[int, int, int]] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to dynamically calculated size.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`dict[str, int]` or `int`):
                The size to resize the image to. If a dictionary, it should have the keys `"height"` and `"width"`.
            background_color (`tuple[int, int, int]`):
                The background color to use for the padding.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BICUBIC`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `None`: will be inferred from input
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        background_color = background_color if background_color is not None else self.background_color
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        height, width = get_image_size(image, input_data_format)
        max_size = max(height, width)

        size = get_size_dict(size, default_to_square=True)
        if size["height"] != size["width"]:
            raise ValueError(
                f"Output height and width must be the same. Got height={size['height']} and width={size['width']}"
            )
        size = size["height"]

        delta = size / max_size
        # Largest side becomes `size` and the other side is scaled according to the aspect ratio.
        output_size_nonpadded = [
            max(int(height * delta), self.min_size),
            max(int(width * delta), self.min_size),
        ]

        image = resize(
            image,
            size=output_size_nonpadded,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
        # Expand and pad the images to obtain a square image of dimensions `size x size`
        image = self.pad_to_square(
            image=image,
            background_color=background_color,
            input_data_format=input_data_format,
        )
        return image

    def postprocess(
        self,
        images: ImageInput,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[list[float]] = None,
        image_std: Optional[list[float]] = None,
        input_data_format: Optional[str] = None,
        return_tensors: Optional[str] = None,
    ):
        """Applies post-processing to the decoded image tokens by reversing transformations applied during preprocessing."""
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = 1.0 / self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        images = make_list_of_images(images)  # Ensures input is a list

        if isinstance(images[0], PIL.Image.Image):
            return images if len(images) > 1 else images[0]

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])  # Determine format dynamically

        pixel_values = []

        for image in images:
            image = to_numpy_array(image)  # Ensure NumPy format

            if do_normalize:
                image = self.unnormalize(
                    image=image, image_mean=image_mean, image_std=image_std, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)
                image = image.clip(0, 255).astype(np.uint8)

            if do_normalize and do_rescale and return_tensors == "PIL.Image.Image":
                image = to_channel_dimension_format(image, ChannelDimension.LAST, input_channel_dim=input_data_format)
                image = PIL.Image.fromarray(image)

            pixel_values.append(image)

        data = {"pixel_values": pixel_values}
        return_tensors = return_tensors if return_tensors != "PIL.Image.Image" else None

        return BatchFeature(data=data, tensor_type=return_tensors)

    def unnormalize(
        self,
        image: np.array,
        image_mean: Union[float, Iterable[float]],
        image_std: Union[float, Iterable[float]],
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.array:
        """
        Unnormalizes `image` using the mean and standard deviation specified by `mean` and `std`.
        image = (image * image_std) + image_mean
        Args:
            image (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)` or `(num_channels, image_size, image_size)`):
                Batch of pixel values to postprocess.
            image_mean (`float` or `Iterable[float]`):
                The mean to use for unnormalization.
            image_std (`float` or `Iterable[float]`):
                The standard deviation to use for unnormalization.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        num_channels = 3

        if isinstance(image_mean, Iterable):
            if len(image_mean) != num_channels:
                raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(image_mean)}")
        else:
            image_mean = [image_mean] * num_channels

        if isinstance(image_std, Iterable):
            if len(image_std) != num_channels:
                raise ValueError(f"std must have {num_channels} elements if it is an iterable, got {len(image_std)}")
        else:
            image_std = [image_std] * num_channels

        rev_image_mean = tuple(-mean / std for mean, std in zip(image_mean, image_std))
        rev_image_std = tuple(1 / std for std in image_std)
        image = self.normalize(
            image=image, mean=rev_image_mean, std=rev_image_std, input_data_format=input_data_format
        )
        return image


__all__ = [
    "JanusImageProcessor",
    "JanusPreTrainedModel",
    "JanusForConditionalGeneration",
    "JanusModel",
    "JanusVQVAE",
    "JanusVisionModel",
    "JanusVQVAEConfig",
    "JanusVisionConfig",
    "JanusConfig",
]
