import torch
from typing import List
from ..mimi import MimiConfig, MimiModel
from dataclasses import dataclass
from ...utils import ModelOutput, auto_docstring


@dataclass
@auto_docstring
class Qwen3TTSTokenizerV2EncoderOutput(ModelOutput):
    r"""
    audio_codes (`List[torch.LongTensor]`):
        Discret code embeddings computed using `model.encode`, each tensor has shape (codes_length_i, num_quantizers).
    """

    audio_codes: List[torch.LongTensor] = None


class Qwen3TTSTokenizerV2Encoder(MimiModel):
    def __init__(self, config: MimiConfig):
        super().__init__(config)
        self.config = config

        self.upsample = None
        self.decoder_transformer = None
        self.decoder = None

        self.post_init()