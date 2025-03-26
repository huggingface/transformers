from ...utils import logging
from ..qwen2.configuration_qwen2 import Qwen2Config

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
