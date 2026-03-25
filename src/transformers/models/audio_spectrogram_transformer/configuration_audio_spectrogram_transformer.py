# Copyright 2022 Google AI and The HuggingFace Inc. team. All rights reserved.
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
"""Audio Spectogram Transformer (AST) model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="MIT/ast-finetuned-audioset-10-10-0.4593")
@strict
class ASTConfig(PreTrainedConfig):
    r"""
    frequency_stride (`int`, *optional*, defaults to 10):
        Frequency stride to use when patchifying the spectrograms.
    time_stride (`int`, *optional*, defaults to 10):
        Temporal stride to use when patchifying the spectrograms.
    max_length (`int`, *optional*, defaults to 1024):
        Temporal dimension of the spectrograms.

    Example:

    ```python
    >>> from transformers import ASTConfig, ASTModel

    >>> # Initializing a AST MIT/ast-finetuned-audioset-10-10-0.4593 style configuration
    >>> configuration = ASTConfig()

    >>> # Initializing a model (with random weights) from the MIT/ast-finetuned-audioset-10-10-0.4593 style configuration
    >>> model = ASTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "audio-spectrogram-transformer"

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    patch_size: int | list[int] | tuple[int, int] = 16
    qkv_bias: bool = True
    frequency_stride: int = 10
    time_stride: int = 10
    max_length: int = 1024
    num_mel_bins: int = 128


__all__ = ["ASTConfig"]
