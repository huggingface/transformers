# Copyright 2020, Microsoft and the HuggingFace Inc. team.
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
"""DeBERTa-v2 model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/deberta-v2-xlarge")
class DebertaV2Config(PreTrainedConfig):
    r"""
    max_relative_positions (`int`, *optional*, defaults to -1):
        The range of relative positions `[-max_position_embeddings, max_position_embeddings]`. Use the same value
        as `max_position_embeddings`.
    pooler_hidden_act (`str`, *optional*, defaults to `"gelu"`):
        Activation function used in the dropout module.
    pooler_dropout (`float`, *optional*, defaults to `0`):
        Dropout rate in the pooler module.
    relative_attention (`bool`, *optional*, defaults to `True`):
        Whether use relative position encoding.
    position_biased_input (`bool`, *optional*, defaults to `True`):
        Whether add absolute position embedding to content embedding.
    pos_att_type (`list[str]`, *optional*):
        The type of relative position attention, it can be a combination of `["p2c", "c2p"]`, e.g. `["p2c"]`,
        `["p2c", "c2p"]`, `["p2c", "c2p"]`.
    legacy (`bool`, *optional*, defaults to `True`):
        Whether or not the model should use the legacy `LegacyDebertaOnlyMLMHead`, which does not work properly
        for mask infilling tasks.

    Example:

    ```python
    >>> from transformers import DebertaV2Config, DebertaV2Model

    >>> # Initializing a DeBERTa-v2 microsoft/deberta-v2-xlarge style configuration
    >>> configuration = DebertaV2Config()

    >>> # Initializing a model (with random weights) from the microsoft/deberta-v2-xlarge style configuration
    >>> model = DebertaV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deberta-v2"

    def __init__(
        self,
        vocab_size=128100,
        hidden_size=1536,
        num_hidden_layers=24,
        num_attention_heads=24,
        intermediate_size=6144,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=0,
        initializer_range=0.02,
        layer_norm_eps=1e-7,
        relative_attention=False,
        max_relative_positions=-1,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
        position_biased_input=True,
        pos_att_type=None,
        pooler_dropout=0,
        pooler_hidden_act="gelu",
        legacy=True,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.relative_attention = relative_attention
        self.max_relative_positions = max_relative_positions
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.position_biased_input = position_biased_input

        # Backwards compatibility
        if isinstance(pos_att_type, str):
            pos_att_type = [x.strip() for x in pos_att_type.lower().split("|")]

        self.pos_att_type = pos_att_type
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps

        self.pooler_hidden_size = kwargs.get("pooler_hidden_size", hidden_size)
        self.pooler_dropout = pooler_dropout
        self.pooler_hidden_act = pooler_hidden_act
        self.legacy = legacy


__all__ = ["DebertaV2Config"]
