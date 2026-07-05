from transformers import (
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
    Gemma4AudioConfig,
)


class Quatfit1TextConfig(Gemma4TextConfig):
    model_type = "quatfit1_text"


class Quatfit1VisionConfig(Gemma4VisionConfig):
    model_type = "quatfit1_vision"


class Quatfit1AudioConfig(Gemma4AudioConfig):
    model_type = "quatfit1_audio"


class Quatfit1Config(Gemma4Config):
    model_type = "quatfit1"

    sub_configs = {
        "text_config": Quatfit1TextConfig,
        "vision_config": Quatfit1VisionConfig,
        "audio_config": Quatfit1AudioConfig,
    }

    def __post_init__(self, **kwargs):

        if isinstance(self.text_config, dict):
            self.text_config = Quatfit1TextConfig(**self.text_config)

        if isinstance(self.vision_config, dict):
            self.vision_config = Quatfit1VisionConfig(**self.vision_config)

        if isinstance(self.audio_config, dict):
            self.audio_config = Quatfit1AudioConfig(**self.audio_config)

        super().__post_init__(**kwargs)