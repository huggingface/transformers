# Copyright 2020 The SqueezeBert authors and The HuggingFace Inc. team.
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
"""SqueezeBERT model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="squeezebert/squeezebert-uncased")
class SqueezeBertConfig(PreTrainedConfig):
    r"""
    q_groups (`int`, *optional*, defaults to 4):
        The number of groups in Q layer.
    k_groups (`int`, *optional*, defaults to 4):
        The number of groups in K layer.
    v_groups (`int`, *optional*, defaults to 4):
        The number of groups in V layer.
    post_attention_groups (`int`, *optional*, defaults to 1):
        The number of groups in the first feed forward network layer.
    intermediate_groups (`int`, *optional*, defaults to 4):
        The number of groups in the second feed forward network layer.
    output_groups (`int`, *optional*, defaults to 4):
        The number of groups in the third feed forward network layer.

    Examples:

    ```python
    >>> from transformers import SqueezeBertConfig, SqueezeBertModel

    >>> # Initializing a SqueezeBERT configuration
    >>> configuration = SqueezeBertConfig()

    >>> # Initializing a model (with random weights) from the configuration above
    >>> model = SqueezeBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "squeezebert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
        embedding_size=768,
        q_groups=4,
        k_groups=4,
        v_groups=4,
        post_attention_groups=1,
        intermediate_groups=4,
        output_groups=4,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.embedding_size = embedding_size
        self.q_groups = q_groups
        self.k_groups = k_groups
        self.v_groups = v_groups
        self.post_attention_groups = post_attention_groups
        self.intermediate_groups = intermediate_groups
        self.output_groups = output_groups


__all__ = ["SqueezeBertConfig"]
