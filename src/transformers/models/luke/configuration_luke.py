# Copyright Studio Ousia and The HuggingFace Inc. team.
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
"""LUKE configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="studio-ousia/luke-base")
class LukeConfig(PreTrainedConfig):
    r"""
    entity_vocab_size (`int`, *optional*, defaults to 500000):
        Entity vocabulary size of the LUKE model. Defines the number of different entities that can be represented
        by the `entity_ids` passed when calling [`LukeModel`].
    entity_emb_size (`int`, *optional*, defaults to 256):
        The number of dimensions of the entity embedding.
    use_entity_aware_attention (`bool`, *optional*, defaults to `True`):
        Whether or not the model should use the entity-aware self-attention mechanism proposed in [LUKE: Deep
        Contextualized Entity Representations with Entity-aware Self-attention (Yamada et
        al.)](https://huggingface.co/papers/2010.01057).

    Examples:

    ```python
    >>> from transformers import LukeConfig, LukeModel

    >>> # Initializing a LUKE configuration
    >>> configuration = LukeConfig()

    >>> # Initializing a model from the configuration
    >>> model = LukeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "luke"

    def __init__(
        self,
        vocab_size=50267,
        entity_vocab_size=500000,
        hidden_size=768,
        entity_emb_size=256,
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
        use_entity_aware_attention=True,
        classifier_dropout=None,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.entity_vocab_size = entity_vocab_size
        self.hidden_size = hidden_size
        self.entity_emb_size = entity_emb_size
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
        self.use_entity_aware_attention = use_entity_aware_attention
        self.classifier_dropout = classifier_dropout


__all__ = ["LukeConfig"]
