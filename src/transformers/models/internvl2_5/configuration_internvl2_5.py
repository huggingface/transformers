# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class InternVL2_5VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternVisionModel`]. It is used to
    instantiate a vision encoder according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            Number of color channels in the input images (e.g., 3 for RGB).
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        qkv_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the queries and values in the self-attention layers.
        hidden_size (`int`, *optional*, defaults to 3200):
            Dimensionality of the encoder layers and the pooler layer.
        num_attention_heads (`int`, *optional*, defaults to 25):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 12800):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        qk_normalization (`bool`, *optional*, defaults to `True`):
            Whether to normalize the queries and keys in the self-attention layers.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        use_flash_attn (`bool`, *optional*, defaults to `True`):
            Whether to use flash attention mechanism.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate for stochastic depth.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 0.1):
            A factor for layer scale.
    """

    model_type = "internvl2_5"
    base_config_key = "vision_config"

    def __init__(
        self,
        num_channels=3,
        patch_size=14,
        image_size=224,
        qkv_bias=False,
        hidden_size=3200,
        num_attention_heads=25,
        intermediate_size=12800,
        qk_normalization=True,
        num_hidden_layers=48,
        hidden_act="gelu",
        norm_type="rms_norm",
        layer_norm_eps=1e-6,
        dropout=0.0,
        drop_path_rate=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.norm_type = norm_type
        self.qkv_bias = qkv_bias
        self.qk_normalization = qk_normalization


class InternVL2_5Config(PretrainedConfig):
    model_type = "internvl2_5"
    sub_configs = {"vision_config": InternVL2_5VisionConfig, "text_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        select_layer=-1,
        hidden_size=None,
        force_image_size=None,
        downsample_ratio=0.5,
        dynamic_image_size=False,
        use_thumbnail=False,
        pixel_shuffle_version="v1",  # pixel_shuffle_version
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        tie_word_embeddings=False,
        image_token_id=151667,
        image_start_token_id=151665,
        image_end_token_id=151666,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = InternVL2_5VisionConfig(**vision_config)
        else:
            self.vision_config = InternVL2_5VisionConfig()

        if isinstance(text_config, dict):
            if text_config["model_type"] in ["qwen2", "llama", "phi3"]:
                self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
            elif text_config["model_type"] == "internlm2":  # InternLM2 not supported in transformers
                internlm2_config = AutoConfig.from_pretrained(text_config["_name_or_path"])
                self.text_config = internlm2_config.from_dict(text_config)
            else:
                raise ValueError(f"Unsupported text model type: {text_config['model_type']}")
        else:
            self.text_config = CONFIG_MAPPING["qwen2"]()

        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.pixel_shuffle_version = pixel_shuffle_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch

        self.hidden_size = hidden_size or self.text_config.hidden_size

        self.image_token_id = image_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id

        # # By default, we use tie_word_embeddings=False for models of all sizes.
        self.tie_word_embeddings = tie_word_embeddings
        self.text_config.tie_word_embeddings = self.tie_word_embeddings
        if tie_word_embeddings != False:
            raise ValueError("InternVL2_5Config: tie_word_embeddings must be False")
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["InternVL2_5Config"]
