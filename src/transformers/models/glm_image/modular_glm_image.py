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

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ..glm4v.configuration_glm4v import Glm4vTextConfig
from ..glm4v.modeling_glm4v import (
    Glm4vForConditionalGeneration,
    Glm4vModel,
    Glm4vTextDecoderLayer,
    Glm4vTextModel,
)
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..siglip.modeling_siglip import (
    SiglipAttention,
    SiglipEncoderLayer,
    SiglipMLP,
    SiglipMultiheadAttentionPoolingHead,
    SiglipPreTrainedModel,
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


class GlmImagePreTrainedModel(SiglipPreTrainedModel):
    config: GlmImageConfig
    base_model_prefix = "model"
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True

    _no_split_modules = [
        "GlmImageVisionEmbeddings",
        "GlmImageVisionBlock",
        "GlmImageVisionMultiheadAttentionPoolingHead",
    ]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    _can_record_outputs = {
        "hidden_states": GlmImageVisionBlock,
        "attentions": GlmImageVisionAttention,
    }

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
            init.normal_(module.fc1.bias, std=1e-6)
            init.normal_(module.fc2.bias, std=1e-6)
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


class GlmImageTextDecoderLayer(Glm4vTextDecoderLayer):
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

        # 计算距离: (z - e)^2 = z^2 + e^2 - 2 * z @ e^T
        # z_flattened: [B*H*W, C], embedding: [N, C]
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2 * torch.matmul(z_flattened, embedding.t())
        )

        # 找到最近的 codebook entry
        indices = torch.argmin(d, dim=1)

        # 获取量化后的向量: [B*H*W, C] -> [B, H, W, C]
        z_q = embedding[indices].view(batch_size, height, width, self.embedding_dim)

        # [B, H, W, C] -> [B, C, H, W]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # indices: [B*H*W] -> [B, H, W]
        indices = indices.view(batch_size, height, width)

        return z_q, indices

    def get_codebook_entry(self, indices: torch.Tensor, bhwc: list) -> torch.Tensor:
        """
        根据索引获取 codebook entry

        Args:
            indices: codebook 索引
            bhwc: 目标形状 [batch, height, width, channel]

        Returns:
            z_q: 量化后的特征，shape: [B, C, H, W]
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
        前向传播

        Args:
            x: 视觉特征，shape: [B, N, C] 其中 N = H * W
            h: 特征图高度
            w: 特征图宽度

        Returns:
            z: 量化后的特征，shape: [B, N, C]
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
        编码为离散 token 索引

        Args:
            x: 视觉特征，shape: [B, N, C]
            h: 特征图高度
            w: 特征图宽度

        Returns:
            indices: codebook 索引，shape: [B, H, W]
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
        从离散 token 索引解码

        Args:
            indices: codebook 索引
            bhwc: 目标形状 [batch, height, width, channel]

        Returns:
            z: 解码后的特征，shape: [B, C, H, W]
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
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
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
    pass


class GlmImageForConditionalGeneration(Glm4vForConditionalGeneration):
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


__all__ = [
    "GlmImageVisionConfig",
    "GlmImageTextConfig",
    "GlmImageConfig",
    "GlmImagePreTrainedModel",
    "GlmImageVisionModel",
    "GlmImageTextModel",
    "GlmImageForConditionalGeneration",
]
