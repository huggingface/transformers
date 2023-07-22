from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

SAFFU_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "saffu-BBLM10M": "https://huggingface.co/saffu-BBLM10M/resolve/main/config.json",
    "saffu-BBLM100M": "https://huggingface.co/saffu-BBLM100M/resolve/main/config.json",
}

class SAFFUConfig(PretrainedConfig):
    model_type = "saffu"
    def __init__(
        self,
        r = 2,
        block_size = 100, 
        heads = 2, 
        N = 4083, # 2**12, 
        bits = 2**8, 
        hidden = 2**9,
        space = False, 
        wave_encode = True,
        **kwargs,
    ):
        self._r = r
        self._block_size = block_size
        self._heads = heads
        self._N = N
        self._bits = bits
        self._hidden = hidden
        self._space = space
        self._wave_encode = wave_encode
        super().__init__(**kwargs)