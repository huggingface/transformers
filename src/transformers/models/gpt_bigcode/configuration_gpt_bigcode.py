# Copyright 2023 The BigCode team and HuggingFace Inc. team.
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
"""GPTBigCode configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/gpt_bigcode")
@strict
class GPTBigCodeConfig(PreTrainedConfig):
    r"""
    scale_attn_weights (`bool`, *optional*, defaults to `True`):
        Scale attention weights by dividing by sqrt(hidden_size)..
    attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
        Whether to call the fused softmax in float32.
    scale_attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
        Whether to scale the attention softmax in float32.
    multi_query (`bool`, *optional*, defaults to `True`):
        Whether to use multi-query attention in the decoder. Ignored when `new_decoder_architecture` is `True`.

    Example:

    ```python
    >>> from transformers import GPTBigCodeConfig, GPTBigCodeModel

    >>> # Initializing a GPTBigCode configuration
    >>> configuration = GPTBigCodeConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = GPTBigCodeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gpt_bigcode"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int | None = None
    activation_function: str = "gelu_pytorch_tanh"
    resid_pdrop: float | int = 0.1
    embd_pdrop: float | int = 0.1
    attn_pdrop: float | int = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    use_cache: bool = True
    bos_token_id: int | None = 50256
    eos_token_id: int | list[int] | None = 50256
    pad_token_id: int | None = None
    attention_softmax_in_fp32: bool = True
    scale_attention_softmax_in_fp32: bool = True
    multi_query: bool = True
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = 1 if self.multi_query else self.n_head
        super().__post_init__(**kwargs)


__all__ = ["GPTBigCodeConfig"]
