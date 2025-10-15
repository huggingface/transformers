from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ..auto import CONFIG_MAPPING, AutoConfig
from ..dac.configuration_dac import DacConfig


class PEAudioEncoderConfig(PretrainedConfig):
    model_type = "pe_audio_encoder"
    sub_configs = {"dac_config": DacConfig}

    def __init__(
        self,
        dac_config=None,
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
        if isinstance(dac_config, dict):
            dac_config = DacConfig.from_dict(dac_config)
        elif dac_config is None:
            dac_config = DacConfig()

        self.dac_config = dac_config

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


class PEAudioConfig(PretrainedConfig):
    model_type = "pe_audio"
    sub_configs = {"text_config": AutoConfig, "audio_config": PEAudioEncoderConfig}

    def __init__(
        self,
        text_config=None,
        audio_config=None,
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

        if isinstance(audio_config, dict):
            audio_config = PEAudioEncoderConfig.from_dict(audio_config)
        elif audio_config is None:
            audio_config = PEAudioEncoderConfig()
            # TODO: add log

        self.text_config = text_config
        self.audio_config = audio_config

        self.projection_dim = projection_dim
        self.nth_text_layer = nth_text_layer


__all__ = ["PEAudioEncoderConfig", "PEAudioConfig"]
