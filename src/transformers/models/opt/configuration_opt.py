# Copyright 2022 The Metaseq Authors and The HuggingFace Inc. team. All rights reserved.
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
"""OPT model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/opt-350m")
class OPTConfig(PreTrainedConfig):
    r"""
    do_layer_norm_before (`bool`, *optional*, defaults to `True`):
        Whether to perform layer normalization before the attention block.
    word_embed_proj_dim (`int`, *optional*):
        `word_embed_proj_dim` can be set to down-project word embeddings, *e.g.* `opt-350m`. Defaults to
        `hidden_size`.
    enable_bias (`bool`, *optional*, defaults to `True`):
        Whether or not if the linear layers in the attention blocks should use the bias term.
    layer_norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
        Whether or not if the layer norms should have learnable parameters.
    _remove_final_layer_norm (`bool`, *optional*, defaults to `False`):
        Whether or not to remove the final layer norm.

    Example:

    ```python
    >>> from transformers import OPTConfig, OPTModel

    >>> # Initializing a OPT facebook/opt-large style configuration
    >>> configuration = OPTConfig()

    >>> # Initializing a model (with random weights) from the facebook/opt-large style configuration
    >>> model = OPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "opt"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50272,
        hidden_size=768,
        num_hidden_layers=12,
        ffn_dim=3072,
        max_position_embeddings=2048,
        do_layer_norm_before=True,
        _remove_final_layer_norm=False,
        word_embed_proj_dim=None,
        dropout=0.1,
        attention_dropout=0.0,
        num_attention_heads=12,
        activation_function="relu",
        layerdrop=0.0,
        init_std=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=2,
        enable_bias=True,
        layer_norm_elementwise_affine=True,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.word_embed_proj_dim = word_embed_proj_dim if word_embed_proj_dim is not None else hidden_size
        self.ffn_dim = ffn_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.layerdrop = layerdrop
        self.use_cache = use_cache
        self.do_layer_norm_before = do_layer_norm_before
        # We keep these variables at `True` for backward compatibility.
        self.enable_bias = enable_bias
        self.layer_norm_elementwise_affine = layer_norm_elementwise_affine

        # Note that the only purpose of `_remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        self._remove_final_layer_norm = _remove_final_layer_norm


__all__ = ["OPTConfig"]
