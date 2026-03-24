# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
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
"""RecurrentGemma model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/recurrentgemma-2b")
@strict
class RecurrentGemmaConfig(PreTrainedConfig):
    r"""
    lru_width (`int` or `None`, *optional*):
        Dimension of the hidden representations of the RG-LRU. If `None`
        this will be set to `hidden_size`.
        Whether to scale the output of the embeddings by `sqrt(hidden_size)`.
    attention_window_size (`int`, *optional*, defaults to 2048):
        The size of the attention window used in the attention block.
    conv1d_width (`int`, *optional*, defaults to 4):
        The kernel size of conv1d layers used in the recurrent blocks.
    logits_soft_cap (`float`, *optional*, defaults to 30.0):
        The value at which the logits should be soft-capped to after the transformer and LM-head computation in the Causal LM architecture.
    block_types (`list[str]`, *optional*, defaults to `('recurrent', 'recurrent', 'attention')`):
        List of aleternating blocks that will be repeated to initialize the `temporal_block` layer.
    w_init_variance_scale (`float`, *optional*, defaults to 0.01):
        weight initialization variance.

    ```python
    >>> from transformers import RecurrentGemmaModel, RecurrentGemmaConfig

    >>> # Initializing a RecurrentGemma recurrentgemma-2b style configuration
    >>> configuration = RecurrentGemmaConfig()

    >>> # Initializing a model from the recurrentgemma-2b style configuration
    >>> model = RecurrentGemmaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "recurrent_gemma"
    attribute_map = {"sliding_window": "attention_window_size"}

    num_hidden_layers: int = 26
    vocab_size: int = 256000
    hidden_size: int = 2560
    intermediate_size: int = 3 * 2560
    num_attention_heads: int = 10
    lru_width: int | None = None
    attention_window_size: int = 2048
    conv1d_width: int = 4
    logits_soft_cap: float = 30.0
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = 2
    hidden_activation: str = "gelu_pytorch_tanh"
    rope_parameters: RopeParameters | dict | None = None
    block_types: list[str] | tuple[str, ...] | None = ("recurrent", "recurrent", "attention")
    attention_dropout: float | int = 0.0
    num_key_value_heads: int | None = None
    attention_bias: bool = False
    w_init_variance_scale: float = 0.01
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.lru_width = self.lru_width if self.lru_width is not None else self.hidden_size
        self.block_types = list(self.block_types)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = (
            self.num_key_value_heads if self.num_key_value_heads is not None else self.num_attention_heads
        )
        self.final_w_init_variance_scale = 2.0 / self.num_hidden_layers
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("The number of `num_key_value_heads` must be smaller than `num_attention_heads`")

    @property
    def layers_block_type(self):
        return (self.block_types * 100)[: self.num_hidden_layers]


__all__ = ["RecurrentGemmaConfig"]
