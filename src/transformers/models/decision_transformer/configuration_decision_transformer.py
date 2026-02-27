# Copyright 2022 The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""Decision Transformer model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="")
class DecisionTransformerConfig(PreTrainedConfig):
    """
    state_dim (`int`, *optional*, defaults to 17):
        The state size for the RL environment
    act_dim (`int`, *optional*, defaults to 4):
        The size of the output action space
    max_ep_len (`int`, *optional*, defaults to 4096):
        The maximum length of an episode in the environment
    action_tanh (`bool`, *optional*, defaults to True):
        Whether to use a tanh activation on action prediction
    scale_attn_weights (`bool`, *optional*, defaults to `True`):
        Scale attention weights by dividing by sqrt(hidden_size)..
    scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
        Whether to additionally scale attention weights by `1 / layer_idx + 1`.
    reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
        Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
        dot-product/softmax to float() when training with mixed precision.

    Example:

    ```python
    >>> from transformers import DecisionTransformerConfig, DecisionTransformerModel

    >>> # Initializing a DecisionTransformer configuration
    >>> configuration = DecisionTransformerConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = DecisionTransformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "decision_transformer"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        state_dim=17,
        act_dim=4,
        hidden_size=128,
        max_ep_len=4096,
        action_tanh=True,
        vocab_size=1,
        n_positions=1024,
        n_layer=3,
        n_head=1,
        n_inner=None,
        activation_function="relu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        add_cross_attention=False,
        **kwargs,
    ):
        self.add_cross_attention = add_cross_attention
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_ep_len = max_ep_len
        self.action_tanh = action_tanh
        self.vocab_size = vocab_size
        self.n_positions = n_positions
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
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(**kwargs)


__all__ = ["DecisionTransformerConfig"]
