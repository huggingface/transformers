from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig


class GraniteSpeechEncoderConfig(PretrainedConfig):
    model_type = "granite_speech_encoder"

    def __init__(
        self,
        input_dim=160,
        num_layers=10,
        hidden_dim=1024,
        feedforward_mult=4,
        num_heads=8,
        dim_head=128,
        output_dim=42,
        context_size=200,
        dropout=0.1,
        conv_kernel_size=15,
        conv_expansion_factor=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.feedforward_mult = feedforward_mult
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.output_dim = output_dim
        self.context_size = context_size
        self.dropout = dropout
        self.conv_kernel_size = conv_kernel_size
        self.conv_expansion_factor = conv_expansion_factor


class GraniteSpeechConfig(PretrainedConfig):
    model_type = "granite_speech"
    sub_configs = {
        "text_config": AutoConfig,
        "encoder_config": GraniteSpeechEncoderConfig,
        "projector_config": AutoConfig,
    }

    def __init__(
        self,
        encoder_config=None,
        text_config=None,
        projector_config=None,
        audio_token_index=49155,
        initializer_range=0.02,
        has_lora_adapter=True,
        # Extra projector stuff
        downsample_rate=5,
        window_size=15,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "granite"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["granite"]()

        if isinstance(projector_config, dict):
            projector_config["model_type"] = (
                projector_config["model_type"] if "model_type" in projector_config else "blip_2_qformer"
            )
            projector_config = CONFIG_MAPPING[projector_config["model_type"]](**projector_config)
        elif projector_config is None:
            projector_config = CONFIG_MAPPING["blip_2_qformer"]()

        if not isinstance(encoder_config, GraniteSpeechEncoderConfig):
            encoder_config = {} if encoder_config is None else encoder_config
            encoder_config = GraniteSpeechEncoderConfig(**encoder_config)

        self.text_config = text_config
        self.encoder_config = encoder_config
        self.projector_config = projector_config
        self.audio_token_index = audio_token_index
        self.initializer_range = initializer_range
        self.has_lora_adapter = has_lora_adapter
        self.downsample_rate = downsample_rate
        self.window_size = window_size
        super().__init__(**kwargs)


__all__ = ["GraniteSpeechEncoderConfig", "GraniteSpeechConfig"]
