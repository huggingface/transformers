# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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


import numpy as np
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="bezzam/xcodec2")
@strict(accept_kwargs=True)
class Xcodec2Config(PreTrainedConfig):
    r"""
    downsampling_ratios (`list[int]`, *optional*, defaults to `[2, 2, 4, 4, 5]`):
        Ratios for downsampling in the encoder.
    semantic_model_config (`Union[Dict, Wav2Vec2BertConfig]`, *optional*):
        An instance of the configuration object for the semantic (Wav2Vec2BertConfig) model.
    quantization_dim (`int`, *optional*, defaults to 2048):
        Dimension for the vector quantization codebook.
    quantization_levels (`list[int]`, *optional*, defaults to `[4, 4, 4, 4, 4, 4, 4, 4]`):
        Levels for the vector quantization codebook.

    Example:

    ```python
    >>> from transformers import Xcodec2Config, Xcodec2Model

    >>> # Initializing configuration
    >>> configuration = Xcodec2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Xcodec2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xcodec2"

    sub_configs = {"semantic_model_config": AutoConfig}

    encoder_hidden_size: int = 48
    downsampling_ratios: list[int] | tuple[int, ...] = (2, 2, 4, 4, 5)
    decoder_hidden_size: int = 1024
    semantic_model_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02
    sampling_rate: int = 16000
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    num_hidden_layers: int = 12
    activation_dropout: float = 0.1
    attention_dropout: float = 0.0
    attention_bias: bool = False
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    head_dim: int = 64
    quantization_dim: int = 2048
    quantization_levels: list[int] | tuple[int, ...] = (4, 4, 4, 4, 4, 4, 4, 4)
    max_position_embeddings: int = 4096
    rope_parameters: dict | None = None

    def __post_init__(self, **kwargs):
        if isinstance(self.semantic_model_config, dict):
            self.semantic_model_config["model_type"] = self.semantic_model_config.get("model_type", "wav2vec2-bert")
            self.semantic_model_config = CONFIG_MAPPING[self.semantic_model_config["model_type"]](
                **self.semantic_model_config
            )
        elif self.semantic_model_config is None:
            self.semantic_model_config = CONFIG_MAPPING["wav2vec2-bert"]()

        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": 10000.0, "rope_type": "default"}

        super().__post_init__(**kwargs)

    @property
    def hop_length(self) -> int:
        return int(np.prod(self.downsampling_ratios))

    @property
    def n_fft(self) -> int:
        return self.hop_length * 4

    @property
    def hidden_size(self) -> int:
        # NOTE: for modular usage of LlamaAttention
        return self.decoder_hidden_size


__all__ = ["Xcodec2Config"]
