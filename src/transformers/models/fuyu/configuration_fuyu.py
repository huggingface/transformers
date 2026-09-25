# Copyright 2023 Adept AI and the HuggingFace Inc. team. All rights reserved.
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
"""Fuyu model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging
from ..auto import AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="adept/fuyu-8b")
@strict
class FuyuConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import FuyuConfig

    >>> # Initializing a Fuyu fuyu-7b style configuration
    >>> configuration = FuyuConfig()
    ```"""

    model_type = "fuyu"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="persimmon",
            init_kwargs={
                "vocab_size": 262144,
                "max_position_embeddings": 16384,
                "hidden_size": 4096,
                "intermediate_size": 16384,
                "num_hidden_layers": 36,
                "num_attention_heads": 64,
                "hidden_act": "relu2",
                "initializer_range": 0.02,
                "layer_norm_eps": 1e-5,
                "use_cache": True,
                "qk_layernorm": True,
                "hidden_dropout": 0.0,
                "attention_dropout": 0.0,
                "eos_token_id": 2,
            },
        ),
    }
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 25000.0

    vocab_size: int = 262144
    hidden_size: int = 4096
    intermediate_size: int = 16384
    num_hidden_layers: int = 36
    num_attention_heads: int = 64
    hidden_act: str = "relu2"
    max_position_embeddings: int = 16384
    image_size: int | None = 300
    patch_size: int | None = 30
    num_channels: int | None = 3
    initializer_range: float = 0.02
    layer_norm_eps: float | None = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    qk_layernorm: bool | None = True
    hidden_dropout: float | int | None = 0.0
    attention_dropout: float | int | None = 0.0
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    image_token_id: int | None = 71011
    text_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC
        super().__post_init__(**kwargs)


__all__ = ["FuyuConfig"]
