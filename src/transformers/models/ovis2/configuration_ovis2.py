from ...configuration_utils import PretrainedConfig
from ..qwen2.configuration_qwen2 import Qwen2Config


class Ovis2VisionConfig(PretrainedConfig):
    r"""

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 2816):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05): <fill_docstring>
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        projection_dropout (`float`, *optional*, defaults to 0.0): <fill_docstring>
        qkv_bias (`bool`, *optional*, defaults to `False`): <fill_docstring>
        use_bias (`bool`, *optional*, defaults to `False`): <fill_docstring>
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        vocab_size (`<fill_type>`, *optional*, defaults to 16384): <fill_docstring>
        hidden_stride (`<fill_type>`, *optional*, defaults to 1): <fill_docstring>
        vision_feature_select_strategy (`<fill_type>`, *optional*, defaults to `"full"`): <fill_docstring>
        num_visual_indicator_tokens (`<fill_type>`, *optional*, defaults to 5): <fill_docstring>
        tokenize_function (`<fill_type>`, *optional*, defaults to `"softmax"`): <fill_docstring>

    Example:

    ```python
    >>> from transformers import Ovis2VisionConfig, Ovis2VisionModel

    >>> # Initializing a Ovis2VisionConfig with google/ovis2-base-patch16-224 style configuration
    >>> configuration = Ovis2VisionConfig()

    >>> # Initializing a Ovis2VisionModel (with random weights) from the google/ovis2-base-patch16-224 style configuration
    >>> model = Ovis2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 8,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 14,
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        qkv_bias: bool = False,
        use_bias: bool = False,
        hidden_act="silu",
        vocab_size=16384,
        hidden_stride=1,
        vision_feature_select_strategy="full",
        num_visual_indicator_tokens=5,
        tokenize_function="softmax",
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
        self.hidden_act = hidden_act
        self.use_bias = use_bias
        self.qkv_bias = qkv_bias
        self.rms_norm_eps = rms_norm_eps
        self.projection_dropout = projection_dropout
        self.vocab_size = vocab_size
        self.hidden_stride = hidden_stride
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.num_visual_indicator_tokens = num_visual_indicator_tokens
        self.tokenize_function = tokenize_function


class Ovis2Config(PretrainedConfig):
    model_type = "ovis2"
    sub_configs = {"text_config": Qwen2Config, "vision_config": Ovis2VisionConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=151665,
        visual_indicator_token_ids=[151666, 151667, 151668, 151669, 151670],
        vocab_size=151643,
        sliding_window=32768,
        hidden_size=1536,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = Ovis2VisionConfig(**vision_config)
        elif isinstance(vision_config, Ovis2VisionConfig):
            self.vision_config = vision_config
        if vision_config is None:
            self.vision_config = Ovis2VisionConfig(num_visual_indicator_tokens=len(visual_indicator_token_ids))

        if isinstance(text_config, dict):
            self.text_config = Qwen2Config(**text_config)
        elif isinstance(text_config, Qwen2Config):
            self.text_config = text_config
        elif text_config is None:
            self.text_config = Qwen2Config()

        self.vocab_size = vocab_size
        self.sliding_window = sliding_window
        self.hidden_size = hidden_size

        self.image_token_id = image_token_id
        self.visual_indicator_token_ids = visual_indicator_token_ids
        super().__init__(**kwargs)


__all__ = ["Ovis2VisionConfig", "Ovis2Config"]
