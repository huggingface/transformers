from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ..auto import CONFIG_MAPPING, AutoConfig
from ..timm_wrapper import TimmWrapperConfig


class PEVideoEncoderConfig(PretrainedConfig):
    model_type = "pe_video_encoder"
    sub_configs = {"vision_encoder_config": TimmWrapperConfig}

    def __init__(
        self,
        vision_encoder_config=None,
        hidden_size=1792,
        intermediate_size=4800,
        num_hidden_layers=28,
        num_attention_heads=14,
        num_key_value_heads=14,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=10000,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        rope_theta=20000,
        rope_scaling=None,
        attention_bias=False,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
    ):
        if isinstance(vision_encoder_config, dict):
            vision_encoder_config = TimmWrapperConfig.from_dict(vision_encoder_config)
        elif vision_encoder_config is None:
            vision_encoder_config = TimmWrapperConfig()

        self.vision_encoder_config = vision_encoder_config

        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(**kwargs)


class PEVideoConfig(PretrainedConfig):
    model_type = "pe_video"
    sub_configs = {"text_config": AutoConfig, "video_config": PEVideoEncoderConfig}

    def __init__(
        self,
        text_config=None,
        video_config=None,
        projection_dim=1024,
        nth_text_layer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "modernbert")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["modernbert"]()
            # TODO: add log

        if isinstance(video_config, dict):
            video_config = PEVideoEncoderConfig.from_dict(video_config)
        elif video_config is None:
            video_config = PEVideoEncoderConfig()
            # TODO: add log

        self.text_config = text_config
        self.video_config = video_config

        self.projection_dim = projection_dim
        self.nth_text_layer = nth_text_layer


__all__ = ["PEVideoEncoderConfig", "PEVideoConfig"]
