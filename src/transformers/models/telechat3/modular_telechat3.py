# Copyright 2026 China Telecom Artificial Intelligence Technology Co., Ltd. and the HuggingFace Inc. team. All rights reserved.
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

import math

from huggingface_hub.dataclasses import strict

from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ..llama.configuration_llama import LlamaConfig


@auto_docstring(checkpoint="Tele-AI/TeleChat3-36B-Thinking")
@strict
class TeleChat3Config(LlamaConfig):
    r"""
    ```python
    >>> from transformers import AutoModelForCausalLM, TeleChat3Config

    >>> # Initializing a TeleChat3-36B style configuration
    >>> configuration = TeleChat3Config()

    >>> # Initializing a model (which uses the Llama implementation) from that configuration
    >>> model = AutoModelForCausalLM.from_config(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "telechat3"

    vocab_size: int = 131072
    hidden_size: int = 6144
    intermediate_size: int = 24576
    num_hidden_layers: int = 64
    num_attention_heads: int = 48
    num_key_value_heads: int | None = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.0048
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 3
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    mlp_bias: bool = False
    head_dim: int | None = 128

    pretraining_tp = AttributeError()

    def convert_rope_params_to_dict(self, **kwargs):
        kwargs = super().convert_rope_params_to_dict(**kwargs)
        # The original checkpoints use a custom `"telechat3-yarn"` RoPE type, which is YaRN with 0.07 instead of 0.1
        # as the coefficient of its default attention factor. Convert it to plain YaRN with an explicit attention factor
        rope_parameters = self.rope_parameters
        if rope_parameters.get("rope_type") == "telechat3-yarn":
            rope_parameters.pop("type", None)
            rope_parameters["rope_type"] = "yarn"
            if "original_max_position_embeddings" in rope_parameters:
                # The original code derives the scaling factor from the context lengths, ignoring `factor`
                rope_parameters["factor"] = (
                    self.max_position_embeddings / rope_parameters["original_max_position_embeddings"]
                )
            else:
                # The base standardization sets this for `"yarn"`, but it saw `"telechat3-yarn"`
                rope_parameters["original_max_position_embeddings"] = self.max_position_embeddings
            if rope_parameters.get("attention_factor") is None:
                factor = rope_parameters["factor"]
                mscale = rope_parameters.get("mscale")
                mscale_all_dim = rope_parameters.get("mscale_all_dim")

                def get_mscale(scale, mscale=1.0):
                    return 1.0 if scale <= 1 else 0.07 * mscale * math.log(scale) + 1.0

                if mscale and mscale_all_dim:
                    attention_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
                else:
                    attention_factor = get_mscale(factor)
                rope_parameters["attention_factor"] = attention_factor
        return kwargs


__all__ = ["TeleChat3Config"]
