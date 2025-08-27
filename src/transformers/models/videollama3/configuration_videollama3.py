"""VideoLLaMA3 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig
from ..qwen2 import Qwen2Config


logger = logging.get_logger(__name__)


class Videollama3VisionConfig(PretrainedConfig):
    """
    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
    """

    model_type = "videollama3_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range


class Videollama3Config(PretrainedConfig):
    """
    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen2Config`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Videollama3VisionConfig`):
            The config object or dictionary of the vision backbone.
        use_token_compression (`bool`, *optional*, defaults to `False`):
            Whether to use temporal token compression to reduce the number of video tokens.
        image_token_id (`int`, *optional*, defaults to -1):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to -1):
            The video token index to encode the image prompt.
    """

    model_type = "videollama3"
    sub_configs = {"vision_config": Videollama3VisionConfig, "text_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        use_token_compression=False,
        image_token_id=-1,
        video_token_id=-1,
        **kwargs,
    ):
        if text_config is None:
            self.text_config = Qwen2Config(**kwargs)
            logger.info("text_config is None, using default qwen2 config")
        elif isinstance(text_config, dict):
            assert "model_type" in text_config, "text_config must contain 'model_type' key"
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif isinstance(text_config, PretrainedConfig):
            self.text_config = text_config
        else:
            raise ValueError(
                "text_config must be a dictionary, PretrainedConfig instance, or None. "
                f"Got {type(text_config)} instead."
            )

        if vision_config is None:
            self.vision_config = Videollama3VisionConfig()
            logger.info("vision_config is None, using default vision config")
        elif isinstance(vision_config, dict):
            assert "model_type" in vision_config, "vision_config must contain 'model_type' key"
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif isinstance(vision_config, PretrainedConfig):
            self.vision_config = vision_config
        else:
            raise ValueError(
                "vision_config must be a dictionary, PretrainedConfig instance, or None. "
                f"Got {type(vision_config)} instead."
            )

        self.use_token_compression = use_token_compression
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        super().__init__(**kwargs)


__all__ = ["Videollama3Config", "Videollama3VisionConfig"]
