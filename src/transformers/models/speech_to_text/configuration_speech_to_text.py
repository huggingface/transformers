# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Speech2Text model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/s2t-small-librispeech-asr")
@strict
class Speech2TextConfig(PreTrainedConfig):
    r"""
    max_source_positions (`int`, *optional*, defaults to 6000):
        The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
    max_target_positions (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model might ever be used with. Typically, set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    num_conv_layers (`int`, *optional*, defaults to 2):
        Number of 1D convolutional layers in the conv module.
    conv_kernel_sizes (`tuple[int]`, *optional*, defaults to `(5, 5)`):
        A tuple of integers defining the kernel size of each 1D convolutional layer in the conv module. The length
        of `conv_kernel_sizes` has to match `num_conv_layers`.
    conv_channels (`int`, *optional*, defaults to 1024):
        An integer defining the number of output channels of each convolution layers except the final one in the
        conv module.
    input_feat_per_channel (`int`, *optional*, defaults to 80):
        An integer specifying the size of feature vector. This is also the dimensions of log-mel filter-bank
        features.

    Example:

    ```python
    >>> from transformers import Speech2TextConfig, Speech2TextModel

    >>> # Initializing a Speech2Text s2t_transformer_s style configuration
    >>> configuration = Speech2TextConfig()

    >>> # Initializing a model (with random weights) from the s2t_transformer_s style configuration
    >>> model = Speech2TextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "speech_to_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "encoder_layers",
    }

    vocab_size: int = 10000
    encoder_layers: int = 12
    encoder_ffn_dim: int = 2048
    encoder_attention_heads: int = 4
    decoder_layers: int = 6
    decoder_ffn_dim: int = 2048
    decoder_attention_heads: int = 4
    encoder_layerdrop: float | int = 0.0
    decoder_layerdrop: float | int = 0.0
    use_cache: bool = True
    is_encoder_decoder: bool = True
    activation_function: str = "relu"
    d_model: int = 256
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    decoder_start_token_id: int = 2
    scale_embedding: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    max_source_positions: int = 6000
    max_target_positions: int = 1024
    num_conv_layers: int = 2
    conv_kernel_sizes: list[int] | tuple[int, ...] = (5, 5)
    conv_channels: int = 1024
    input_feat_per_channel: int = 80
    input_channels: int = 1
    tie_word_embeddings: bool = True

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if len(self.conv_kernel_sizes) != self.num_conv_layers:
            raise ValueError(
                "Configuration for convolutional module is incorrect. "
                "It is required that `len(config.conv_kernel_sizes)` == `config.num_conv_layers` "
                f"but is `len(config.conv_kernel_sizes) = {len(self.conv_kernel_sizes)}`, "
                f"`config.num_conv_layers = {self.num_conv_layers}`."
            )


__all__ = ["Speech2TextConfig"]
