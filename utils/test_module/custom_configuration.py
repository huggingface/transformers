from transformers import PretrainedConfig


class CustomConfig(PretrainedConfig):
    model_type = "custom"

    def __init__(self, attribute=1, **kwargs):
        self.attribute = attribute
        super().__init__(**kwargs)


class NoSuperInitConfig(PretrainedConfig):
    model_type = "custom"

    def __init__(self, attribute=1, **kwargs):
        self.attribute = attribute
