# coding=utf-8
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
""" RecurrentGemma model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


from ..deprecated._archive_maps import GEMMA_PRETRAINED_CONFIG_ARCHIVE_MAP  # noqa: F401, E402


_PATTERN = ("recurrent", "recurrent", "attention")


class RecurrentGemmaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RecurrentGemmaModel`]. It is used to instantiate an RecurrentGemma
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the RecurrentGemma-7B.

    e.g. [google/recurrentgemma-2b](https://huggingface.co/google/recurrentgemma-2b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        num_hidden_layers (`<fill_type>`, *optional*, defaults to 26): <fill_docstring>
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the RecurrentGemma model. Defines the number of
            different tokens that can be represented by the
            `inputs_ids` passed when calling [`RecurrentGemmaModel`]
        hidden_size (`int`, *optional*, defaults to 2560):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 7680):
            Dimension of the MLP representations.
        num_attention_heads (`int`, *optional*, defaults to 10):
            The number of heads for the attention block and the number of
            heads/blocks for the block-diagonal layers used in the RG-LRU gates.
            This number must divide `hidden_size` and `lru_width`.
        lru_width (`int` or `None`, *optional*):
            Dimension of the hidden representations of the RG-LRU. If `None`
            this will be set to `hidden_size`.
            Whether to scale the output of the embeddings by `sqrt(hidden_size)`.
        embeddings_scale_by_sqrt_dim (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        attention_window_size (`int`, *optional*, defaults to 2048):
            The size of the attention window used in the attention block.
        conv1d_width (`<fill_type>`, *optional*, defaults to 4): <fill_docstring>
        logits_soft_cap (`<fill_type>`, *optional*, defaults to 30.0): <fill_docstring>
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values
            attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        hidden_activation (`<fill_type>`, *optional*, defaults to `"gelu_pytorch_tanh"`): <fill_docstring>
        partial_rotary_factor (`<fill_type>`, *optional*, defaults to 0.5): <fill_docstring>
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        block_types (`<fill_type>`, *optional*, defaults to `('recurrent', 'recurrent', 'attention')`): <fill_docstring>
        attention_dropout (`<fill_type>`, *optional*, defaults to 0.0): <fill_docstring>
        num_key_value_heads (`<fill_type>`, *optional*, defaults to 16): <fill_docstring>
        attention_bias (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
    ```python
    >>> from transformers import RecurrentGemmaModel, RecurrentGemmaConfig

    >>> # Initializing a RecurrentGemma recurrentgemma-2b style configuration
    >>> configuration = RecurrentGemmaConfig()

    >>> # Initializing a model from the recurrentgemma-2b style configuration
    >>> model = RecurrentGemmaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "recurrent_gemma"
    # keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        num_hidden_layers=26,
        vocab_size=256000,
        hidden_size=2560,
        intermediate_size=3 * 2560,
        num_attention_heads=10,
        lru_width=None,
        embeddings_scale_by_sqrt_dim=True,
        attention_window_size=2048,
        conv1d_width=4,
        logits_soft_cap=30.0,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        hidden_activation="gelu_pytorch_tanh",
        partial_rotary_factor=0.5,
        rope_theta=10000.0,
        block_types=("recurrent", "recurrent", "attention"),
        attention_dropout=0.0,
        num_key_value_heads=16,
        attention_bias=False,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.lru_width = lru_width if lru_width is not None else hidden_size
        self.embeddings_scale_by_sqrt_dim = embeddings_scale_by_sqrt_dim
        self.attention_window_size = attention_window_size
        self.conv1d_width = conv1d_width
        self.logits_soft_cap = logits_soft_cap
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self._block_types = list(block_types)
        self.hidden_activation = hidden_activation
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def block_types(self) -> tuple[str, ...]:
        return (self._block_types * 100)[: self.num_hidden_layers]
