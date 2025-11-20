# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""S3Tokenizer model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class S3TokenizerConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`S3TokenizerModel`]. It is used to instantiate a
    S3Tokenizer model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        n_mels (`int`, *optional*, defaults to 128):
            Number of mel-frequency bins for the mel-spectrogram.
        n_fft (`int`, *optional*, defaults to 400):
            Size of the FFT window for computing the mel-spectrogram.
        hop_length (`int`, *optional*, defaults to 160):
            Number of audio samples between adjacent STFT columns (10ms at 16kHz).
        token_rate (`int`, *optional*, defaults to 25):
            Number of speech tokens generated per second of audio (25 Hz for v2 models).
        vocab_size (`int`, *optional*, defaults to 6561):
            Vocabulary size of the S3 tokenizer (3^8 for FSQ quantization).
        n_audio_ctx (`int`, *optional*, defaults to 1500):
            Maximum audio context length.
        n_audio_state (`int`, *optional*, defaults to 1280):
            Hidden state dimension of the audio encoder.
        n_audio_head (`int`, *optional*, defaults to 20):
            Number of attention heads in the audio encoder.
        n_audio_layer (`int`, *optional*, defaults to 6):
            Number of transformer layers in the audio encoder.
        use_sdpa (`bool`, *optional*, defaults to `False`):
            Whether to use Scaled Dot Product Attention (SDPA) for faster inference.

    Example:

    ```python
    >>> from transformers import S3TokenizerModel, S3TokenizerConfig

    >>> # Initializing a S3Tokenizer configuration
    >>> configuration = S3TokenizerConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = S3TokenizerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "s3tokenizer"

    def __init__(
        self,
        sampling_rate=16000,
        n_mels=128,
        n_fft=400,
        hop_length=160,
        token_rate=25,
        vocab_size=6561,
        n_audio_ctx=1500,
        n_audio_state=1280,
        n_audio_head=20,
        n_audio_layer=6,
        use_sdpa=False,
        **kwargs,
    ):
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.token_rate = token_rate
        self.vocab_size = vocab_size
        self.n_audio_ctx = n_audio_ctx
        self.n_audio_state = n_audio_state
        self.n_audio_head = n_audio_head
        self.n_audio_layer = n_audio_layer
        self.use_sdpa = use_sdpa
        # Add hidden_size as an alias for n_audio_state for compatibility with common tests
        self.hidden_size = n_audio_state

        super().__init__(**kwargs)


__all__ = ["S3TokenizerConfig"]
