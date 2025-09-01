"""AudioFlamingo3 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class AudioFlamingo3EncoderConfig(PretrainedConfig):
    model_type = "audioflamingo3_encoder"

    def __init__(
        self,
        num_mel_bins: int = 128,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        encoder_layerdrop: float = 0.0,
        d_model: int = 1280,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_function: str = "gelu",
        activation_dropout: float = 0.0,
        scale_embedding: bool = False,
        initializer_range: float = 0.02,
        max_source_positions: int = 1500,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.max_source_positions = max_source_positions


class AudioFlamingo3Config(PretrainedConfig):
    model_type = "audioflamingo3"
    sub_configs = {"llm_cfg": AutoConfig, "sound_tower_cfg": AutoConfig}

    def __init__(
        self,
        llm_cfg=None,
        sound_tower_cfg=None,
        sound_mm_projector_cfg=None,
        **kwargs,
    ) -> None:

        if isinstance(sound_tower_cfg, dict):
            sound_tower_cfg["model_type"] = sound_tower_cfg.get("model_type", "audioflamingo3_encoder")
            sound_tower_cfg = CONFIG_MAPPING[sound_tower_cfg["model_type"]](**sound_tower_cfg)
        elif sound_tower_cfg is None:
            sound_tower_cfg = CONFIG_MAPPING["audioflamingo3_encoder"](
                d_model=1280,
                encoder_attention_heads=20,
                encoder_ffn_dim=5120,
                encoder_layerdrop=0.0,
                encoder_layers=32,
                num_mel_bins=128,
                max_source_positions=1500,
                scale_embedding=False,
                activation_function="gelu",
            )

        self.sound_tower_cfg = sound_tower_cfg

        if isinstance(llm_cfg, dict):
            llm_cfg["model_type"] = llm_cfg.get("model_type", "qwen2")
            llm_cfg = CONFIG_MAPPING[llm_cfg["model_type"]](**llm_cfg)
        elif llm_cfg is None:
            llm_cfg = CONFIG_MAPPING["qwen2"]()

        self.llm_cfg = llm_cfg

        self.sound_mm_projector_cfg = sound_mm_projector_cfg
        super().__init__(**kwargs)


__all__ = ["AudioFlamingo3Config", "AudioFlamingo3EncoderConfig"]
