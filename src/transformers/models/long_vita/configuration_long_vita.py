from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import Qwen2Config


logger = logging.get_logger(__name__)



class LongVITAConfig(Qwen2Config):
    model_type = "long_vita"

    def __init__(
        self,
        **kwargs,
    ):

        super().__init__(
            **kwargs,
        )
