# Copyright 2026 IQuest Lab and the HuggingFace Inc. team. All rights reserved.
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


@auto_docstring(checkpoint="IQuestLab/IQuest-Coder-V1-40B-Instruct")
@strict
class IQuestCoderConfig(LlamaConfig):
    r"""
    ```python
    >>> from transformers import AutoModelForCausalLM, IQuestCoderConfig

    >>> # Initializing an IQuest-Coder-V1-40B style configuration
    >>> configuration = IQuestCoderConfig()

    >>> # Initializing a model (which uses the Llama implementation) from that configuration
    >>> model = AutoModelForCausalLM.from_config(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "iquestcoder"

    vocab_size: int = 76800
    hidden_size: int = 5120
    intermediate_size: int = 27648
    num_hidden_layers: int = 80
    num_attention_heads: int = 40
    num_key_value_heads: int | None = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
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
        # The original code supports these options, but no released checkpoint enables them
        if getattr(self, "clip_qkv", None) is not None:
            raise ValueError("IQuestCoder does not support `clip_qkv`.")
        if getattr(self, "use_sliding_window", False):
            raise ValueError("IQuestCoder does not support `use_sliding_window`.")


__all__ = ["IQuestCoderConfig"]
