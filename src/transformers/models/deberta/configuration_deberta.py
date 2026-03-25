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
"""DeBERTa model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/deberta-base")
@strict
class DebertaConfig(PreTrainedConfig):
    r"""
    relative_attention (`bool`, *optional*, defaults to `False`):
        Whether use relative position encoding.
    max_relative_positions (`int`, *optional*, defaults to -1):
        The range of relative positions `[-max_position_embeddings, max_position_embeddings]`. Use the same value
        as `max_position_embeddings`.
    position_biased_input (`bool`, *optional*, defaults to `True`):
        Whether add absolute position embedding to content embedding.
    pos_att_type (`list[str]`, *optional*):
        The type of relative position attention, it can be a combination of `["p2c", "c2p"]`, e.g. `["p2c"]`,
        `["p2c", "c2p"]`.
    pooler_dropout (`float`, *optional*, defaults to `0`):
        Dropout rate in the pooler module.
    pooler_hidden_act (`str`, *optional*, defaults to `"gelu"`):
        Activation function used in the dropout module.
    legacy (`bool`, *optional*, defaults to `True`):
        Whether or not the model should use the legacy `LegacyDebertaOnlyMLMHead`, which does not work properly
        for mask infilling tasks.

    Example:

    ```python
    >>> from transformers import DebertaConfig, DebertaModel

    >>> # Initializing a DeBERTa microsoft/deberta-base style configuration
    >>> configuration = DebertaConfig()

    >>> # Initializing a model (with random weights) from the microsoft/deberta-base style configuration
    >>> model = DebertaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deberta"

    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-7
    relative_attention: bool = False
    max_relative_positions: int = -1
    pad_token_id: int | None = 0
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    position_biased_input: bool = True
    pos_att_type: str | list[str] | None = None
    pooler_dropout: float | int = 0.0
    pooler_hidden_act: str = "gelu"
    legacy: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        # Backwards compatibility
        if isinstance(self.pos_att_type, str):
            self.pos_att_type = [x.strip() for x in self.pos_att_type.lower().split("|")]

        self.pooler_hidden_size = kwargs.get("pooler_hidden_size", self.hidden_size)
        super().__post_init__(**kwargs)


__all__ = ["DebertaConfig"]
