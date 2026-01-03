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

from collections.abc import Callable
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationMixin
from ...image_processing_utils import get_size_dict
from ...image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import (
    TensorType,
    TransformersKwargs,
    is_torchdynamo_compiling,
    logging,
)
from ..chameleon.modeling_chameleon import ChameleonVQVAEVectorQuantizer
from ..glm4v.configuration_glm4v import Glm4vTextConfig
from ..glm4v.modeling_glm4v import (
    Glm4vCausalLMOutputWithPast,
    Glm4vModel,
    Glm4vModelOutputWithPast,
    Glm4vPreTrainedModel,
    Glm4vTextAttention,
    Glm4vTextDecoderLayer,
    Glm4vTextModel,
    Glm4vTextRotaryEmbedding,
)
from ..glm4v_moe.modeling_glm4v_moe import apply_multimodal_rotary_pos_emb, eager_attention_forward
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..siglip.image_processing_siglip import SiglipImageProcessor
from ..siglip.modeling_siglip import (
    SiglipAttention,
    SiglipEncoderLayer,
    SiglipMLP,
    SiglipVisionEmbeddings,
    default_flax_embed_init,
    lecun_normal_,
)


logger = logging.get_logger(__name__)


class GlmImageVQVAEConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmImageVQModel`]. It is used to instantiate a
    `GlmImageVQModel` according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information. Instantiating a
    configuration with the defaults will yield a similar configuration to the VQModel of the
    [zai-org/GLM-Image](https://huggingface.co/zai-org/GLM-Image) architecture.

    Args:
        embed_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of each embedding vector.
        num_embeddings (`int`, *optional*, defaults to 16384):
            Number of codebook embeddings.
        latent_channels (`int`, *optional*, defaults to 1536):
            Number of channels for the latent space.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        base_channels (`int`, *optional*, defaults to 128):
            Base channel count.
        num_res_blocks (`int`, *optional*, defaults to 2):
            Number of residual blocks.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """

    model_type = "glm_image_vqmodel"
    base_config_key = "vq_config"

    def __init__(
        self,
        embed_dim: int = 2048,
        num_embeddings: int = 16384,
        latent_channels: int = 1536,
        in_channels: int = 3,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_embeddings = num_embeddings
        self.latent_channels = latent_channels
        self.in_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.initializer_range = initializer_range


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
    sub_configs = {
        "vision_config": GlmImageVisionConfig,
        "text_config": GlmImageTextConfig,
        "vq_config": GlmImageVQVAEConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        vq_config=None,
        image_token_id=167855,
        image_start_token_id=167851,
        image_end_token_id=167852,
        **kwargs,
    ):
        if isinstance(vq_config, dict):
            self.vq_config = self.sub_configs["vq_config"](**vq_config)
        elif vq_config is None:
            self.vq_config = self.sub_configs["vq_config"](**kwargs)

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


class GlmImageTextAttention(Glm4vTextAttention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    and "Generating Long Sequences with Sparse Transformers".
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(  # diff with Llama
            query_states, key_states, cos, sin, self.rope_parameters["mrope_section"]
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

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
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GlmImageTextRotaryEmbedding(Glm4vTextRotaryEmbedding):
    pass


class GlmImageTextDecoderLayer(Glm4vTextDecoderLayer):
    pass


class GlmImagePreTrainedModel(Glm4vPreTrainedModel):
    config: GlmImageConfig
    input_modalities = ("image", "text")

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, GlmImageTextRotaryEmbedding):
            config = module.config
            base = config.rope_parameters["rope_theta"]
            partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 0.5)
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
            init.copy_(module.inv_freq, inv_freq)
            init.copy_(module.original_inv_freq, inv_freq)
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


class GlmImageVisionResnetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.activate = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=channels)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.activate(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.activate(hidden_states)
        hidden_states = self.conv2(hidden_states)
        return hidden_states + residual


class GlmImageVectorQuantizer(ChameleonVQVAEVectorQuantizer):
    def forward(self, hidden_state: torch.Tensor):
        hidden_state = hidden_state.permute(0, 2, 3, 1).contiguous()
        hidden_state_flattened = hidden_state.view(-1, self.embedding_dim)

        # L2 normalize
        hidden_state = F.normalize(hidden_state, p=2, dim=-1)
        hidden_state_flattened = F.normalize(hidden_state_flattened, p=2, dim=-1)
        embedding = F.normalize(self.embedding.weight, p=2, dim=-1)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        distances = (
            torch.sum(hidden_state_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", hidden_state_flattened, embedding.transpose(0, 1))
        )

        min_encoding_indices = torch.argmin(distances, dim=1)
        hidden_state_quant = embedding[min_encoding_indices].view(hidden_state.shape)

        # compute loss for embedding
        loss = torch.mean((hidden_state_quant.detach() - hidden_state) ** 2) + self.beta * torch.mean(
            (hidden_state_quant - hidden_state.detach()) ** 2
        )

        # preserve gradients
        hidden_state_quant = hidden_state + (hidden_state_quant - hidden_state).detach()

        # reshape back to match original input shape
        hidden_state_quant = hidden_state_quant.permute(0, 3, 1, 2).contiguous()

        return hidden_state_quant, loss, min_encoding_indices


class GlmImageVQVAE(GlmImagePreTrainedModel):
    config: GlmImageVQVAEConfig
    _no_split_modules = [
        "GlmImageVectorQuantizer",
        "GlmImageVisionResnetBlock",
    ]

    def __init__(self, config: GlmImageVQVAEConfig):
        super().__init__(config)

        self.quantize = GlmImageVectorQuantizer(config)
        self.quant_conv = nn.Conv2d(config.latent_channels, config.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.embed_dim, config.latent_channels, 1)
        self.post_conv = nn.Sequential(
            *[GlmImageVisionResnetBlock(config.latent_channels) for _ in range(config.num_res_blocks)]
        )

    def encode(self, hidden_states):
        hidden_states = self.quant_conv(hidden_states)
        quant, emb_loss, indices = self.quantize(hidden_states)
        return quant, emb_loss, indices


class GlmImageVisionModel(GlmImagePreTrainedModel):
    config: GlmImageVisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _no_split_modules = ["GlmImageVisionBlock"]

    def __init__(self, config: GlmImageVisionConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = GlmImageVisionEmbeddings(config)
        self.blocks = nn.ModuleList([GlmImageVisionBlock(config) for _ in range(config.num_hidden_layers)])
        self.post_init()

    def forward(
        self,
        pixel_values,
        interpolate_pos_encoding: Optional[bool] = True,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                attention_mask,
                **kwargs,
            )

        return hidden_states


class GlmImageTextModel(Glm4vTextModel):
    pass


class GlmImageModel(Glm4vModel):
    def __init__(self, config):
        super().__init__(config)
        self.visual = GlmImageVisionModel._from_config(config.vision_config)
        self.language_model = GlmImageTextModel._from_config(config.text_config)
        self.vqmodel = GlmImageVQVAE._from_config(config.vq_config)

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

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        interpolate_pos_encoding: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        """
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`GlmImageVisionModel`].

        """
        vision_outputs = self.visual(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        return vision_outputs

    def get_image_tokens(
        self,
        hidden_states: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
    ) -> torch.LongTensor:
        """
        Tokenizes image features into discrete tokens with VQVAE module.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                The image features from vision encoder. seq_len = H * W (number of patches).
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
                The temporal, height and width of feature shape of each image.
                For single images, temporal is typically 1.

        Returns:
            image_tokens (`torch.LongTensor` of shape `(batch_size, num_tokens)`):
                Discrete token indices from the VQVAE codebook.
        """
        batch_size = hidden_states.shape[0]
        _, grid_h, grid_w = image_grid_thw[0].tolist()

        hidden_states = hidden_states.transpose(1, 2).contiguous()
        hidden_states = hidden_states.view(batch_size, -1, grid_h, grid_w)

        _, _, image_toks = self.vqmodel.encode(hidden_states)

        return image_toks

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        image_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """
        Replace image placeholder tokens in input_ids with actual image token ids from VQVAE.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Input token ids with image placeholders.
            image_ids (`torch.LongTensor` of shape `(num_images, num_tokens_per_image)` or flattened):
                Discrete token indices from the VQVAE codebook.

        Returns:
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Input token ids with image placeholders replaced by actual image tokens.
        """

        special_image_mask = input_ids == self.config.image_token_id
        n_placeholder_tokens = special_image_mask.sum().item()

        image_ids_flat = image_ids.view(-1)
        n_image_tokens = image_ids_flat.shape[0]

        if n_placeholder_tokens != n_image_tokens:
            raise ValueError(
                f"Number of image placeholder tokens ({n_placeholder_tokens}) does not match "
                f"number of image tokens from VQVAE ({n_image_tokens})"
            )

        input_ids = input_ids.clone()
        input_ids[special_image_mask] = image_ids_flat.to(input_ids.device)

        return input_ids

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

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values)
            image_ids = self.get_image_tokens(image_embeds, image_grid_thw)
            input_ids = self.get_placeholder_mask(input_ids, image_ids)

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

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


class GlmImageImageProcessor(SiglipImageProcessor):
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        short_size: int = 384,
        long_size: int = 1152,
        patch_size: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            **kwargs,
        )
        self.short_size = short_size
        self.long_size = long_size
        self.patch_size = patch_size

    def _get_anyres_size(self, height: int, width: int) -> tuple[int, int]:
        target_w, target_h = float(width), float(height)

        ss = min(target_w, target_h)
        if ss < self.short_size:
            scale = self.short_size / ss
            target_w *= scale
            target_h *= scale

        ls = max(target_w, target_h)
        if ls > self.long_size:
            scale = self.long_size / ls
            target_w *= scale
            target_h *= scale

        target_w = int(round(target_w / self.patch_size)) * self.patch_size
        target_h = int(round(target_h / self.patch_size)) * self.patch_size

        return target_h, target_w

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        do_convert_rgb: Optional[bool] = None,
    ):
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, param_name="size", default_to_square=False)
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = self.fetch_images(images)
        images = make_flat_list_of_images(images)

        if not valid_images(images):
            raise ValueError("Invalid image type")

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once("It looks like you are trying to rescale already rescaled images.")

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])

        # Track image grid sizes for each image
        image_grid_thw = []

        if do_resize:
            resized_images = []
            for image in images:
                if input_data_format == ChannelDimension.FIRST:
                    _, h, w = image.shape
                else:
                    h, w, _ = image.shape
                target_h, target_w = self._get_anyres_size(h, w)
                resized_images.append(
                    resize(
                        image=image, size=(target_h, target_w), resample=resample, input_data_format=input_data_format
                    )
                )
                grid_h = target_h // self.patch_size
                grid_w = target_w // self.patch_size

                # grid_t should be 1 for image
                image_grid_thw.append([1, grid_h, grid_w])
            images = resized_images
        else:
            # If no resize, calculate grid from original image sizes
            for image in images:
                if input_data_format == ChannelDimension.FIRST:
                    _, h, w = image.shape
                else:
                    h, w, _ = image.shape
                grid_h = h // self.patch_size
                grid_w = w // self.patch_size

                # grid_t should be 1 for image
                image_grid_thw.append([1, grid_h, grid_w])

        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                for image in images
            ]

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        data = {
            "pixel_values": images,
            "image_grid_thw": image_grid_thw,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


class GlmImageProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
    }


class GlmImageProcessor(ProcessorMixin):
    r"""
    Constructs a GLM-Image processor which wraps a GLM-Image image processor and a GLM-Image tokenizer into a single processor.
    [`~GlmImageProcessor.__call__`] and [`~GlmImageProcessor.decode`] for more information.
    Args:
        image_processor ([`GlmImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = "<|image|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        **kwargs: Unpack[GlmImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizerFast.__call__`] if `text` is not `None` to encode
        the text.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            GlmImageProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        if image_grid_thw is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    grid = image_grid_thw[index]
                    num_image_tokens = int(grid[1] * grid[2])
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)


__all__ = [
    "GlmImageVQVAEConfig",
    "GlmImageVisionConfig",
    "GlmImageTextConfig",
    "GlmImageConfig",
    "GlmImagePreTrainedModel",
    "GlmImageVQVAE",
    "GlmImageVisionModel",
    "GlmImageTextModel",
    "GlmImageModel",
    "GlmImageForConditionalGeneration",
    "GlmImageImageProcessor",
    "GlmImageProcessor",
]
