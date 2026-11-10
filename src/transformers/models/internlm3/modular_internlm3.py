# Copyright 2026 Shanghai AI Laboratory and the HuggingFace Inc. team. All rights reserved.
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


@auto_docstring(checkpoint="internlm/internlm3-8b-instruct")
@strict
class InternLM3Config(LlamaConfig):
    r"""
    ```python
    >>> from transformers import AutoModelForCausalLM, InternLM3Config

    >>> # Initializing an InternLM3-8B style configuration
    >>> configuration = InternLM3Config()

    >>> # Initializing a model (which uses the Llama implementation) from that configuration
    >>> model = AutoModelForCausalLM.from_config(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "internlm3"
    attribute_map = {"qkv_bias": "attention_bias", "bias": "mlp_bias"}

    vocab_size: int = 128512
    hidden_size: int = 4096
    intermediate_size: int = 10240
    num_hidden_layers: int = 48
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 2
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    mlp_bias: bool = False
    head_dim: int | None = 128

    pretraining_tp = AttributeError()

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        super().validate_architecture()
        if self.attention_bias != self.mlp_bias:
            raise ValueError(
                "InternLM3 with `qkv_bias` different from `bias` is not supported, because the Llama implementation "
                "uses the same bias setting for the query, key, value and output projections."
            )


__all__ = ["InternLM3Config"]
