# Copyright 2026 The LG AI Research and HuggingFace Inc. team. All rights reserved.
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

from huggingface_hub.dataclasses import strict

from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ..llama.configuration_llama import LlamaConfig


@auto_docstring(checkpoint="LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")
@strict
class ExaoneConfig(LlamaConfig):
    r"""
    ```python
    >>> from transformers import AutoModelForCausalLM, ExaoneConfig

    >>> # Initializing an EXAONE-3.0-7.8B style configuration
    >>> configuration = ExaoneConfig()

    >>> # Initializing a model (which uses the Llama implementation) from that configuration
    >>> model = AutoModelForCausalLM.from_config(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "exaone"
    attribute_map = {
        "num_layers": "num_hidden_layers",
        "layer_norm_epsilon": "rms_norm_eps",
        "activation_function": "hidden_act",
    }
    default_theta = 500_000.0

    vocab_size: int = 102400
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 361
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_dropout: float | int = 0.0
    head_dim: int | None = None

    pretraining_tp = AttributeError()


__all__ = ["ExaoneConfig"]
