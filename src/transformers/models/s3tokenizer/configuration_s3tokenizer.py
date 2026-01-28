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
    S3Tokenizer model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the S3Tokenizer
    [ResembleAI/s3tokenizer-v2](https://huggingface.co/ResembleAI/s3tokenizer-v2) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
            sampling_rate (`int`, *optional*, defaults to 16000):
                The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
            n_mels (`int`, *optional*, defaults to 128):
                Number of mel-frequency bins for the mel-spectrogram.
            n_fft (`int`, *optional*, defaults to 400):
                Size of the FFT window for computing the mel-spectrogram.
            vocab_size (`int`, *optional*, defaults to 6561):
                Vocabulary size of the S3 tokenizer (3^8 for FSQ quantization).
            n_audio_state (`int`, *optional*, defaults to 1280):
                Hidden state dimension of the audio encoder.
            n_audio_head (`int`, *optional*, defaults to 20):
                Number of attention heads in the audio encoder.
            n_audio_layer (`int`, *optional*, defaults to 6):
                Number of transformer layers in the audio encoder.
            use_sdpa (`bool`, *optional*, defaults to `False`):
                Whether to use Scaled Dot Product Attention (SDPA) for faster inference.
            num_attention_heads (`<fill_type>`, *optional*): <fill_docstring>
            num_key_value_heads (`<fill_type>`, *optional*): <fill_docstring>
            attention_bias (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
            attention_dropout (`<fill_type>`, *optional*, defaults to 0.0): <fill_docstring>
            max_position_embeddings (`<fill_type>`, *optional*, defaults to 2048): <fill_docstring>
            rope_theta (`<fill_type>`, *optional*, defaults to 10000.0): <fill_docstring>
            rope_scaling (`<fill_type>`, *optional*): <fill_docstring>

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
        vocab_size=6561,
        n_audio_state=1280,
        n_audio_head=20,
        n_audio_layer=6,
        use_sdpa=False,
        num_attention_heads=None,
        num_key_value_heads=None,
        attention_bias=False,
        attention_dropout=0.0,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rope_scaling=None,
        **kwargs,
    ):
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.vocab_size = vocab_size
        self.n_audio_state = n_audio_state
        self.n_audio_head = n_audio_head
        self.n_audio_layer = n_audio_layer
        self.use_sdpa = use_sdpa
        # Add hidden_size as an alias for n_audio_state for compatibility with common tests
        self.hidden_size = n_audio_state
        self.num_attention_heads = num_attention_heads if num_attention_heads is not None else n_audio_head
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else self.num_attention_heads
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        super().__init__(**kwargs)
        self.convert_rope_params_to_dict()


__all__ = ["S3TokenizerConfig"]
