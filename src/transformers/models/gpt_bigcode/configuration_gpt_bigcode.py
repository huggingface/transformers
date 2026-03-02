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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/gpt_bigcode")
class GPTBigCodeConfig(PreTrainedConfig):
    r"""
    multi_query (`bool`, *optional*, defaults to `True`):
        Whether to use multi-query attention in the decoder. Ignored when `new_decoder_architecture` is `True`.
    scale_attn_weights (`bool`, *optional*, defaults to `True`):
        Scale attention weights by dividing by sqrt(hidden_size)..
    attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
        Whether to call the fused softmax in float32.
    scale_attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
        Whether to scale the attention softmax in float32.
    attention_type (`bool`, *optional*, defaults to `True`):
        Whether to use Multi-Query Attion (`True`) or Multi-Head Attention (`False`).

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

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_pytorch_tanh",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=None,
        attention_softmax_in_fp32=True,
        scale_attention_softmax_in_fp32=True,
        multi_query=True,
        add_cross_attention=False,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.add_cross_attention = add_cross_attention
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = scale_attention_softmax_in_fp32
        self.multi_query = multi_query
        self.num_key_value_heads = 1 if multi_query else n_head

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        super().__init__(**kwargs)


__all__ = ["GPTBigCodeConfig"]
