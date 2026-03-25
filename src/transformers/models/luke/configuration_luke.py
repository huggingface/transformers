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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="studio-ousia/luke-base")
@strict
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

    vocab_size: int = 50267
    entity_vocab_size: int = 500000
    hidden_size: int = 768
    entity_emb_size: int = 256
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    use_entity_aware_attention: bool = True
    classifier_dropout: float | int | None = None
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = True


__all__ = ["LukeConfig"]
