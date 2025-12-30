# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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

from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...modeling_rope_utils import RopeParameters
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, is_torchdynamo_compiling
from ..glm4v.configuration_glm4v import Glm4vTextConfig
from ..glm4v.modeling_glm4v import (
    Glm4vCausalLMOutputWithPast,
    Glm4vModel,
    Glm4vModelOutputWithPast,
    Glm4vTextDecoderLayer,
    Glm4vTextModel,
    Glm4vPreTrainedModel,
)
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..siglip.modeling_siglip import (
    SiglipAttention,
    SiglipEncoderLayer,
    SiglipMLP,
    SiglipMultiheadAttentionPoolingHead,
    SiglipVisionEmbeddings,
    default_flax_embed_init,
    lecun_normal_,
)


class GlmImageVisionConfig(SiglipVisionConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmImageVisionModel`]. It is used to instantiate a
    GlmImage vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the GLM-Image
    [zai-org/GLM-Image](https://huggingface.co/zai-org/GLM-Image) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 40):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        image_size (`int`, *optional*, defaults to 2048):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        vq_codebook_size (`int`, *optional*, defaults to 16384):
            The size of the VQ codebook.
        vq_codebook_dim (`int`, *optional*, defaults to 2048):
            The dimension of the VQ codebook embeddings.
        vq_num_conv_layers (`int`, *optional*, defaults to 2):
            The number of convolutional layers in the VQ projector.
         spatial_merge_size (`int`, *optional*, defaults to 2):
            The size used for merging spatial dimensions.
    Example:

    ```python
    >>> from transformers import GlmImageVisionConfig, GlmImageVisionModel

    >>> # Initializing a GlmImageVisionConfig with google/glm_image-base-patch16-224 style configuration
    >>> configuration = GlmImageVisionConfig()

    >>> # Initializing a GlmImageVisionModel (with random weights) from the google/glm_image-base-patch16-224 style configuration
    >>> model = GlmImageVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm_image_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1536,
        intermediate_size=6144,
        num_hidden_layers=40,
        num_attention_heads=24,
        num_channels=3,
        image_size=2048,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        vq_codebook_size=16384,
        vq_codebook_dim=2048,
        vq_num_conv_layers=2,
        spatial_merge_size=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.vq_codebook_size = vq_codebook_size
        self.vq_codebook_dim = vq_codebook_dim
        self.vq_num_conv_layers = vq_num_conv_layers
        self.spatial_merge_size = spatial_merge_size


class GlmImageTextConfig(Glm4vTextConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmImageModel`]. It is used to instantiate a
    GLM-Image model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    GLM-Image [zai-org/GLM-Image](https://huggingface.co/zai-org/GLM-Image).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 168064):
            Vocabulary size of the GlmImage model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GlmImageModel`]
        vision_vocab_size (`int`, *optional*, defaults to 16512):
            Vision vocabulary size of the GlmImage model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`GlmImageVisionModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 13696):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.

    ```python
    >>> from transformers import GlmImageTextModel, GlmImageConfig

    >>> # Initializing a GlmImageConfig style configuration
    >>> configuration = GlmImageConfig()

    >>> # Initializing a model from the GlmImageConfig style configuration
    >>> model = GlmImageTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm_image_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `GlmImage`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_up_proj": "colwise_rep",  # we need to replicate here due to the `chunk` operation
        "layers.*.mlp.down_proj": "rowwise_rep",  # we need to replicate here due to the `chunk` operation
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: Optional[int] = 168064,
        vision_vocab_size: Optional[int] = 16512,
        hidden_size: Optional[int] = 4096,
        intermediate_size: Optional[int] = 13696,
        num_hidden_layers: Optional[int] = 40,
        num_attention_heads: Optional[int] = 32,
        num_key_value_heads: Optional[int] = 2,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 32768,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[int] = 1e-05,
        use_cache: Optional[bool] = True,
        tie_word_embeddings: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.vision_vocab_size = vision_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters

        super().__init__(
            tie_word_embeddings=tie_word_embeddings, ignore_keys_at_rope_validation={"mrope_section"}, **kwargs
        )


class GlmImageConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GLM-Image`]. It is used to instantiate a
    GLM-Image model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    GLM-Image [zai-org/GLM-Image](https://huggingface.co/zai-org/GLM-Image) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `GlmImageTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `GlmImageVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 167855):
            The image token index to encode the image prompt.
        image_start_token_id (`int`, *optional*, defaults to 167851):
            The image start token index to encode the start of image.
        image_end_token_id (`int`, *optional*, defaults to 167852):
            The image end token index to encode the end of image.

    ```python
    >>> from transformers import Glm4vForConditionalGeneration, Glm4vConfig

    >>> # Initializing a GLM-Image style configuration
    >>> configuration = Glm4vConfig()

    >>> # Initializing a model from the GLM-Image style configuration
    >>> model = Glm4vForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm_image"
    sub_configs = {"vision_config": GlmImageVisionConfig, "text_config": GlmImageTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=167855,
        image_start_token_id=167851,
        image_end_token_id=167852,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id

        super().__init__(**kwargs)


class GlmImageVisionMLP(SiglipMLP):
    pass


class GlmImageVisionAttention(SiglipAttention):
    pass


class GlmImageVisionBlock(SiglipEncoderLayer):
    def __init__(self, config: GlmImageVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = GlmImageVisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = GlmImageVisionMLP(config)


class GlmImageTextDecoderLayer(Glm4vTextDecoderLayer):
    pass


class GlmImagePreTrainedModel(Glm4vPreTrainedModel):
    config: GlmImageConfig
    input_modalities = ("image", "text")


    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, GlmImageVisionEmbeddings):
            width = (
                self.config.vision_config.hidden_size
                if isinstance(self.config, GlmImageConfig)
                else self.config.hidden_size
            )
            init.normal_(module.position_embedding.weight, std=1 / np.sqrt(width))
            if hasattr(module, "position_ids"):
                init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
        elif isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)
        elif isinstance(module, GlmImageVisionAttention):
            init.xavier_uniform_(module.q_proj.weight)
            init.xavier_uniform_(module.k_proj.weight)
            init.xavier_uniform_(module.v_proj.weight)
            init.xavier_uniform_(module.out_proj.weight)
            init.zeros_(module.q_proj.bias)
            init.zeros_(module.k_proj.bias)
            init.zeros_(module.v_proj.bias)
            init.zeros_(module.out_proj.bias)
        elif isinstance(module, GlmImageVisionMLP):
            init.xavier_uniform_(module.fc1.weight)
            init.xavier_uniform_(module.fc2.weight)
            init.normal_(module.fc1.bias, std=1e-5)
            init.normal_(module.fc2.bias, std=1e-5)
        elif isinstance(module, GlmImageVisionMultiheadAttentionPoolingHead):
            init.xavier_uniform_(module.probe)
            init.xavier_uniform_(module.attention.in_proj_weight)
            init.zeros_(module.attention.in_proj_bias)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            lecun_normal_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)



class GlmImageModelOutputWithPast(Glm4vModelOutputWithPast):
    pass


class GlmImageVisionEmbeddings(SiglipVisionEmbeddings):
    pass


class GlmImageVisionMultiheadAttentionPoolingHead(SiglipMultiheadAttentionPoolingHead):
    def __init__(self, config: GlmImageVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GlmImageVisionMLP(config)


class GlmImageVisionResidualBlock(nn.Module):
    """
    Implementation of the GLM-Image residual block.
    """

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding="same")
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.activate = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding="same")
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x + res


class GlmImageVisionIBQ(nn.Module):
    """
    Index-Based Quantization Module of GLM-Image
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, l2_norm: bool = True):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.l2_norm = l2_norm

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（推理模式）

        Args:
            z: 输入特征，shape: [B, C, H, W]

        Returns:
            z_q: 量化后的特征，shape: [B, C, H, W]
            indices: codebook 索引，shape: [B, H, W]
        """
        batch_size, channels, height, width = z.shape

        # [B, C, H, W] -> [B, H, W, C] -> [B*H*W, C]
        z_permuted = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z_permuted.view(-1, self.embedding_dim)

        # L2 normalization
        if self.l2_norm:
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        # Calculate distance: (z - e)^2 = z^2 + e^2 - 2 * z @ e^T
        # z_flattened: [B*H*W, C], embedding: [N, C]
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2 * torch.matmul(z_flattened, embedding.t())
        )

        # find the nearest codebook entry
        indices = torch.argmin(d, dim=1)

        # Get IBQ: [B*H*W, C] -> [B, H, W, C]
        z_q = embedding[indices].view(batch_size, height, width, self.embedding_dim)

        # [B, H, W, C] -> [B, C, H, W]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # indices: [B*H*W] -> [B, H, W]
        indices = indices.view(batch_size, height, width)

        return z_q, indices

    def get_codebook_entry(self, indices: torch.Tensor, bhwc: list) -> torch.Tensor:
        """
        Get codebook entries by indices

        Args:
            indices: Codebook indices
            bhwc: Target shape [batch, height, width, channel]

        Returns:
            z_q: Quantized features, shape: [B, C, H, W]
        """
        z_q = self.embedding(indices)

        if bhwc is not None:
            batch_size, height, width, channels = bhwc
            # [B, H, W, C] -> [B, C, H, W]
            z_q = z_q.view(batch_size, height, width, channels)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class GlmImageVisionVQProjector(nn.Module):
    """
    VQ Conv Projector of GLM-Image
    """

    def __init__(
        self,
        in_channels: int = 1536,
        codebook_size: int = 16384,
        codebook_dim: int = 2048,
        num_conv_layers: int = 2,
        num_groups: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.codebook_dim = codebook_dim
        self.quant_conv = nn.Conv2d(in_channels, codebook_dim, kernel_size=1)
        self.quantize = GlmImageVisionIBQ(codebook_size, codebook_dim, l2_norm=True)
        self.post_quant_conv = nn.Conv2d(codebook_dim, in_channels, kernel_size=1)
        self.post_conv = nn.Sequential(
            *[GlmImageVisionResidualBlock(in_channels, num_groups) for _ in range(num_conv_layers)]
        )

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Visual features, shape: [B, N, C] where N = H * W
            h: Feature map height
            w: Feature map width

        Returns:
            z: Quantized features, shape: [B, N, C]
        """
        batch_size, seq_len, channels = x.shape

        # [B, N, C] -> [B, H, W, C] -> [B, C, H, W]
        x = x.view(batch_size, h, w, channels)
        x = x.permute(0, 3, 1, 2).contiguous()

        # 量化
        z = self.quant_conv(x)
        z_q, _ = self.quantize(z)

        # 后处理
        z = self.post_quant_conv(z_q)
        z = self.post_conv(z)

        # [B, C, H, W] -> [B, H, W, C] -> [B, N, C]
        z = z.permute(0, 2, 3, 1).contiguous()
        z = z.view(batch_size, seq_len, channels)

        return z

    def encode(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Encode to discrete token indices

        Args:
            x: Visual features, shape: [B, N, C]
            h: Feature map height
            w: Feature map width

        Returns:
            indices: Codebook indices, shape: [B, H, W]
        """
        batch_size, seq_len, channels = x.shape

        # [B, N, C] -> [B, H, W, C] -> [B, C, H, W]
        x = x.view(batch_size, h, w, channels)
        x = x.permute(0, 3, 1, 2).contiguous()

        z = self.quant_conv(x)
        _, indices = self.quantize(z)
        return indices

    def decode(self, indices: torch.Tensor, bhwc: list) -> torch.Tensor:
        """
        Decode from discrete token indices

        Args:
            indices: Codebook indices
            bhwc: Target shape [batch, height, width, channel]

        Returns:
            z: Decoded features, shape: [B, C, H, W]
        """
        z_q = self.quantize.get_codebook_entry(indices, bhwc)
        z = self.post_quant_conv(z_q)
        z = self.post_conv(z)
        return z


class GlmImageVisionModel(GlmImagePreTrainedModel):
    config: GlmImageVisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image",)

    def __init__(self, config: GlmImageVisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = GlmImageVisionEmbeddings(config)
        self.blocks = nn.ModuleList([GlmImageVisionBlock(config) for _ in range(config.num_hidden_layers)])
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = GlmImageVisionMultiheadAttentionPoolingHead(config)

        self.vq_projector = GlmImageVisionVQProjector(
            in_channels=config.hidden_size,
            codebook_size=config.vq_codebook_size,
            codebook_dim=config.vq_codebook_dim,
            num_conv_layers=config.vq_num_conv_layers,
        )

        self.post_init()

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = self.post_conv_layernorm(hidden_states)

        rotary_pos_emb, image_type_ids = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        hidden_states = self.embeddings(hidden_states, seqlens, grid_thw, image_type_ids[:, 0], image_type_ids[:, 1])

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = hidden_states.view(
            -1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1]
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.config.out_hidden_size)

        hidden_states = self.merger(hidden_states)
        return hidden_states


class GlmImageTextModel(Glm4vTextModel):
    pass


class GlmImageModel(Glm4vModel):
    def __init__(self, config):
        super().__init__(config)
        self.visual = GlmImageVisionModel._from_config(config.vision_config)
        self.language_model = GlmImageTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index for image generation task.

        Explanation:
            For image generation, the input sequence contains only text tokens (the prompt).
            Vision tokens are generated autoregressively by the model during decoding.

            For the text prompt (prefill stage), all three dimensions share the same position IDs,
            identical to standard LLM rotary position embedding.

            Examples:
                input_ids: [T T T T T], here T is for text prompt.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids:   [0, 1, 2, 3, 4]
                width position_ids:    [0, 1, 2, 3, 4]

            For the generated vision tokens (decode stage), we use 2D spatial position encoding.
            The temporal dimension is fixed at `gen_st_idx` (the position after the last text token),
            while height and width dimensions follow a row-major 2D grid layout.

            Examples:
                Assuming prompt_length = 5, generated image latent size: height = 2, width = 3
                gen_st_idx = 5 (the position where vision generation starts)

                Generated vision tokens layout (row-major order):
                [V0, V1, V2, V3, V4, V5] representing a 2x3 grid:
                    V0(0,0)  V1(0,1)  V2(0,2)
                    V3(1,0)  V4(1,1)  V5(1,2)

                temporal position_ids: [5, 5, 5, 5, 5, 5]  (all fixed at gen_st_idx)
                height position_ids:   [5, 5, 5, 6, 6, 6]  (gen_st_idx + row_index)
                width position_ids:    [5, 6, 7, 5, 6, 7]  (gen_st_idx + col_index)

            Complete sequence example (prompt + generated vision):
                input_ids: [T T T T T V V V V V V]
                temporal position_ids: [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5]
                height position_ids:   [0, 1, 2, 3, 4, 5, 5, 5, 6, 6, 6]
                width position_ids:    [0, 1, 2, 3, 4, 5, 6, 7, 5, 6, 7]

        Note:
            This function only handles the prefill stage (text prompt).
            The decode stage position IDs are calculated in `prepare_inputs_for_generation`.
            The `_gen_st_idx` attribute is saved here for use during decoding.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary (text prompt only).
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of the generated image's latent feature shape.
                For image generation, temporal is typically 1, and we use height and width
                to determine the 2D grid layout.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
                - 1 for tokens that are **not masked**
                - 0 for tokens that are **masked**

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`):
                Position IDs for temporal, height, and width dimensions.
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size, 1)`):
                The difference between the maximum position and sequence length,
                used for position calculation in subsequent decoding steps.
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]

        # For text-only input, all three dimensions share the same positions: [0, 1, 2, ..., seq_length-1]
        position_ids = torch.arange(seq_length, device=device).view(1, 1, -1).expand(3, batch_size, -1).clone()

        # Save gen_st_idx for decode stage
        # This is where vision token generation starts
        if attention_mask is not None:
            valid_lengths = attention_mask.sum(dim=1)
            self._gen_st_idx = valid_lengths[0].item()
        else:
            self._gen_st_idx = seq_length

        # mrope_position_deltas for pure text input
        mrope_position_deltas = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        return position_ids, mrope_position_deltas

    def get_video_features(self):
        """
        Not Using now
        """
        return None

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
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
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        return special_image_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, GlmImageModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        return GlmImageModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


class GlmImageCausalLMOutputWithPast(Glm4vCausalLMOutputWithPast):
    pass


class GlmImageForConditionalGeneration(GlmImagePreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = GlmImageModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vision_vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, GlmImageCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, GlmImageForConditionalGeneration

        >>> model = GlmImageForConditionalGeneration.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")
        >>> processor = AutoProcessor.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return GlmImageCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        image_grid_thw=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # GLM-Image position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None

        device = input_ids.device
        batch_size = input_ids.shape[0]
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if past_length == 0:
            self._prompt_length = input_ids.shape[1]
            self._gen_latent_h = image_grid_thw[-1, 1].item()
            self._gen_latent_w = image_grid_thw[-1, 2].item()
        else:
            gen_st_idx = self.model._gen_st_idx
            generated_vision_count = past_length - self._prompt_length
            h_idx = generated_vision_count // self._gen_latent_w
            w_idx = generated_vision_count % self._gen_latent_w
            position_ids = torch.tensor(
                [
                    [[gen_st_idx]],
                    [[gen_st_idx + h_idx]],
                    [[gen_st_idx + w_idx]],
                ],
                dtype=torch.long,
                device=device,
            ).expand(-1, batch_size, -1)

            model_inputs["position_ids"] = position_ids

        return model_inputs

    def _get_image_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get the number of images for each sample.

        Returns:
            image_counts (`torch.LongTensor` of shape `(batch_size,)`)
        """
        if inputs_embeds is not None:
            image_token_embed = self.get_input_embeddings()(
                torch.tensor(self.config.image_start_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            is_image = (inputs_embeds == image_token_embed).all(dim=-1)
        else:
            is_image = input_ids == self.config.image_start_token_id

        return is_image.sum(dim=1)

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            image_nums = self._get_image_nums(input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None))

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


__all__ = [
    "GlmImageVisionConfig",
    "GlmImageTextConfig",
    "GlmImageConfig",
    "GlmImagePreTrainedModel",
    "GlmImageVisionModel",
    "GlmImageTextModel",
    "GlmImageForConditionalGeneration",
]
