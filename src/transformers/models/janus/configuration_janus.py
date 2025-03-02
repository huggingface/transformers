# coding=utf-8
# Copyright 2025 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
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
"""Janus model configuration"""

from typing import List

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class JanusVisionConfig(PretrainedConfig):
    """Encoder Vision config in this case its the SIGLIP model"""

    model_type = "siglip_vision_model"
    base_config_key = "encoder_vision_config"

    def __init__(
        self,
        hidden_size=1024,
        mlp_ratio=4.0,
        projection_dim=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=384,
        patch_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        qkv_bias=True,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        initializer_range=0.02,
        select_feature="same",
        select_layer=-1,
        num_register_tokens=0,
        hidden_dropout_rate=0.0,
        projection_dropout=0.0,
        use_qk_norm=False,
        layerscale_value=None,
        use_vision_head=True,
        aligner_projection_size=2048,
        depth=2,
        num_image_tokens=576,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.qkv_bias = qkv_bias
        self.attention_dropout = attention_dropout
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.intermediate_size = int(hidden_size * mlp_ratio)
        self.num_register_tokens = num_register_tokens
        self.hidden_dropout_rate = hidden_dropout_rate
        self.projection_dropout = projection_dropout
        self.use_qk_norm = use_qk_norm
        self.initializer_range = initializer_range
        self.layerscale_value = layerscale_value
        self.select_layer = select_layer
        self.select_feature = select_feature
        self.use_vision_head = use_vision_head
        self.use_special_tokens = kwargs.get("use_special_tokens", False)
        self.depth = depth
        self.aligner_projection_size = aligner_projection_size
        self.num_image_tokens = num_image_tokens


class JanusVQVAEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`JanusVQModel`]. It is used to instantiate a
    `JanusVQModel` according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information. Instantiating a
    configuration with the defaults will yield a similar configuration to the VQModel of the
    [yaswanthgali/Janus-Pro-1B-HF](https://huggingface.co/yaswanthgali/Janus-Pro-1B-HF).

    Args:
        embed_dim (`int`, *optional*, defaults to 256):
            Dimensionality of each embedding vector.
        num_embeddings (`int`, *optional*, defaults to 8192):
            Number of codebook embeddings.
        double_latent (`bool`, *optional*, defaults to `False`):
            Whether to use double z channels.
        latent_channels (`int`, *optional*, defaults to 256):
            Number of channels for the latent space.
        num_patches (`int`, *optional*, defaults to 512):
            Num of patches the input images can be divided into.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        base_channels (`int`, *optional*, defaults to 128):
            Base channel count.
        channel_multiplier (`List[int]`, *optional*, defaults to `[1, 1, 2, 2, 4]`):
            Channel multipliers for each resolution.
        num_res_blocks (`int`, *optional*, defaults to 2):
            Number of residual blocks.
        attn_resolutions (`List[int]`, *optional*):
            Resolutions to apply attention.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate.
        attn_type (`str`, *optional*, defaults to `"vanilla"`):
            Attention type used in VQ-GAN encoder. Can be "vanilla" or None.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """

    model_type = "janus_vqgan"
    base_config_key = "vq_config"

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
        channel_multiplier: List[int] = [1, 1, 2, 2, 4],
        num_res_blocks: int = 2,
        attn_resolutions: List[int] = None,
        dropout: float = 0.0,
        attn_type: str = "vanilla",
        initializer_range=0.02,
        aligner_projection_size=2048,
        depth=2,
        hidden_act="gelu",
        image_token_embed_size=2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_embeddings = num_embeddings
        self.double_latent = double_latent
        self.latent_channels = latent_channels
        self.num_patches = num_patches
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_multiplier = channel_multiplier
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.attn_type = attn_type
        self.initializer_range = initializer_range
        self.out_channels = out_channels
        self.aligner_projection_size = aligner_projection_size
        self.depth = depth
        self.hidden_act = hidden_act
        self.image_token_embed_size = image_token_embed_size


class JanusConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`JanusModel`]. It is used to instantiate an
    Janus model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Janus-1B or Janus-7B models.

    e.g. [yaswanthgali/Janus-Pro-1B-HF](https://huggingface.co/yaswanthgali/Janus-Pro-1B-HF)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 100581):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`.
        vision_feature_layer (`Union[int, List[int]]`, *optional*, defaults to -2):
            The index of the layer to select the vision feature. If multiple indices are provided,
            the vision feature of the corresponding indices will be concatenated to form the
            vision features.
        image_seq_length (`int`, *optional*, defaults to 576):
            Sequence length of one image embedding.
        multimodal_projector_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the multimodal projector.

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

    def __init__(self, text_config=None, vision_config=None, vq_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)

        elif text_config is None:
            logger.info("`text_config` is None. Initializing with default values")
            self.text_config = CONFIG_MAPPING["llama"](
                hidden_size=2048,
                num_hidden_layers=24,
                num_attention_heads=16,
                intermediate_size=5632,
                max_position_embeddings=16384,
                num_key_value_heads=16,
                vocab_size=102400,
            )
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

        # This dimension is required when decoding discrete image tokens to continuous input.
        self.vq_config.num_patches = self.vision_config.image_size // self.vision_config.patch_size
        self.image_token_index = kwargs.get("image_token_index", 100581)
        super().__init__(**kwargs)


__all__ = ["JanusVQVAEConfig", "JanusVisionConfig", "JanusConfig"]
