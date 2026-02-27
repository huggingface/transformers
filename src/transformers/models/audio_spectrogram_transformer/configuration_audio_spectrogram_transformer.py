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

from typing import Any

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="MIT/ast-finetuned-audioset-10-10-0.4593")
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

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        patch_size=16,
        qkv_bias=True,
        frequency_stride=10,
        time_stride=10,
        max_length=1024,
        num_mel_bins=128,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.patch_size = patch_size
        self.qkv_bias = qkv_bias
        self.frequency_stride = frequency_stride
        self.time_stride = time_stride
        self.max_length = max_length
        self.num_mel_bins = num_mel_bins

    # Overwritten from the parent class: AST is not compatible with `generate`, but has a config parameter sharing the
    # same name (`max_length`). Sharing the same name triggers checks regarding the config -> generation_config
    # generative parameters deprecation cycle, overwriting this function prevents this from happening.
    def _get_non_default_generation_parameters(self) -> dict[str, Any]:
        return {}


__all__ = ["ASTConfig"]
