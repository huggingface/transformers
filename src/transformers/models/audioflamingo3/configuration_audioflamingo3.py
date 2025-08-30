"""AudioFlamingo3 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class LlavaConfig(PretrainedConfig):
    model_type = "llava_llama"

    def __init__(
        self,
        llm_cfg=None,
        sound_tower_cfg=None,
        sound_mm_projector_cfg=None,
        model_dtype=None,
        hidden_size=None,
        media_tokens=None,
        ignore_index=None,
        sound_hidden_size=None,
        sound_encoder: str = "",
        **kwargs,
    ):
        super().__init__()
        self.llm_cfg = llm_cfg
        self.sound_tower_cfg = sound_tower_cfg
        self.sound_mm_projector_cfg = sound_mm_projector_cfg
        self.model_dtype = model_dtype
        self.hidden_size = hidden_size
        self.media_tokens = media_tokens
        self.ignore_index = ignore_index
        self.sound_hidden_size = sound_hidden_size
        self.sound_encoder = sound_encoder


class AudioFlamingo3Config(PretrainedConfig):
    model_type = "audioflamingo3"
    attribute_map = {
        "audio_token_id": "audio_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "audio_config": AutoConfig}

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_index=151646,
        **kwargs,
    ):
        self.audio_token_index = audio_token_index

        if isinstance(audio_config, dict):
            audio_config["model_type"] = audio_config.get("model_type", "audioflamingo3_encoder")
            audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif audio_config is None:
            audio_config = CONFIG_MAPPING["audioflamingo3_encoder"](
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

        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "qwen2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"]()

        self.text_config = text_config

        super().__init__(**kwargs)


class AudioFlamingo3EncoderConfig(PretrainedConfig):
    model_type = "audioflamingo3_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        encoder_layerdrop=0.0,
        d_model=1280,
        dropout=0.0,
        attention_dropout=0.0,
        activation_function="gelu",
        activation_dropout=0.0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=1500,
        **kwargs,
    ):
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
