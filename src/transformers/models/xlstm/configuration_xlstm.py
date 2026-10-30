# Copyright 2025 NXAI GmbH. All rights reserved.
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

"""xLSTM configuration."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


def round_up_to_next_multiple_of(x: int, multiple_of: int) -> int:
    """Rounds up x to the next multiple of multiple_of."""
    return int(((x + multiple_of - 1) // multiple_of) * multiple_of)


@auto_docstring(checkpoint="NX-AI/xLSTM-7b")
@strict
class xLSTMConfig(PreTrainedConfig):
    r"""
    num_heads (`int`, *optional*, defaults to 8):
        Number of heads of the mLSTM layers.
    norm_eps (`float`, *optional*, defaults to 1e-06):
        Epsilon of the normalization layers.
    qk_dim_factor (`float`, *optional*, defaults to 0.5):
        Size of the query and key projections, as a fraction of `hidden_size` rounded up to a multiple of 64.
    v_dim_factor (`float`, *optional*, defaults to 1.0):
        Size of the value projection, as a fraction of `hidden_size` rounded up to a multiple of 64.
    ffn_proj_factor (`float`, *optional*, defaults to 2.667):
        Size of the feed forward network, as a fraction of `hidden_size`.
    ffn_round_up_to_multiple_of (`int`, *optional*, defaults to 64):
        The size of the feed forward network is rounded up to a multiple of this value.
    gate_soft_cap (`float`, *optional*, defaults to 15.0):
        Soft cap of the input and forget gate pre-activations.
    output_logit_soft_cap (`float`, *optional*, defaults to 30.0):
        Soft cap of the output logits.

    Example:

    ```python
    >>> from transformers import xLSTMConfig, xLSTMModel

    >>> # Initializing a xLSTM configuration
    >>> configuration = xLSTMConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = xLSTMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xlstm"
    # the original configs name them `embedding_dim` and `num_blocks`
    attribute_map = {"embedding_dim": "hidden_size", "num_blocks": "num_hidden_layers"}

    vocab_size: int = 50304
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_heads: int = 8
    norm_eps: float = 1e-6
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    ffn_proj_factor: float = 2.667
    ffn_round_up_to_multiple_of: int = 64
    gate_soft_cap: float = 15.0
    output_logit_soft_cap: float = 30.0
    tie_word_embeddings: bool = False
    use_cache: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2

    @property
    def qk_dim(self) -> int:
        return round_up_to_next_multiple_of(self.hidden_size * self.qk_dim_factor, multiple_of=64)

    @property
    def v_dim(self) -> int:
        return round_up_to_next_multiple_of(self.hidden_size * self.v_dim_factor, multiple_of=64)

    @property
    def qk_head_dim(self) -> int:
        return self.qk_dim // self.num_heads

    @property
    def v_head_dim(self) -> int:
        return self.v_dim // self.num_heads


__all__ = ["xLSTMConfig"]
