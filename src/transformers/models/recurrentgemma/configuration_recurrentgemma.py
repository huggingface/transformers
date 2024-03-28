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
        block_types: ('Sequence[str]`, *optional*)
            A sequence containing the type of the residual blocks in the
            architecture, specifying each block in order if it should use a
            recurrent or an attention sub-block for the temporal-mixing.
        vocab_size (`int`, *optional*, defaults to 256128):
            Vocabulary size of the RecurrentGemma model. Defines the number of
            different tokens that can be represented by the
            `inputs_ids` passed when calling [`RecurrentGemmaModel`]
        width (`int`, *optional*, defaults to 2560):
            Dimension of the hidden representations.
        mlp_expnaded_width (`int`, *optional*, defaults to 24576):
            Dimension of the MLP representations.
        num_heads (`int`, *optional*, defaults to 10):
            The number of heads for the attention block and the number of
            heads/blocks for the block-diagonal layers used in the RG-LRU gates.
            This number must divide `width` and `lru_width`.
        lru_width (`int` or `None`, *optional*, defaults to None):
            Dimension of the hidden representations of the RG-LRU. If `None`
            this will be set to `width`.
        embeddings_scale_by_sqrt_dim (`bool`, defaults to `True`, *optional*,
        defaults to `True`):
            Whether to scale the output of the embeddings by `sqrt(width)`.
        attention_window_size (`int`, *optional*, defaults to 2048)
            The size of the attention window used in the attention block.
        logits_soft_cap: (`float`, *optional*, defaults to 30.0)
            This will cap the values of the final logits to not exceed this cap
            in absolute value by applying a `tanh`.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for
            initializing all weight matrices.
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
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
    ```python
    >>> from transformers import RecurrentGemmaModel, RecurrentGemmaConfig

    >>> # Initializing a RecurrentGemma recurrentgemma-2b style configuration
    >>> configuration = RecurrentGemmaConfig()

    >>> # Initializing a model from the recurrentgemma-2b style configuration
    >>> model = RecurrentGemmaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
        self,
        num_hidden_layers=26,
        vocab_size=256000,
        width=2560,
        mlp_expanded_width=3 * 2560,
        num_heads=10,
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
        rope_theta=10000.0,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.width = width
        self.mlp_expanded_width = mlp_expanded_width
        self.num_heads = num_heads
        self.num_attention_heads = num_heads
        self.head_dim = self.width // self.num_heads
        self.lru_width = lru_width if lru_width is not None else width
        self.embeddings_scale_by_sqrt_dim = embeddings_scale_by_sqrt_dim
        self.attention_window_size = attention_window_size
        self.conv1d_width = conv1d_width
        self.logits_soft_cap = logits_soft_cap
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def block_types(self) -> tuple[str, ...]:
        return (_PATTERN * 100)[:self.num_hidden_layers]
