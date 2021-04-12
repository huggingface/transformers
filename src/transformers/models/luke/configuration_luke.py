# coding=utf-8
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
""" LUKE configuration """

from ...utils import logging
from ..roberta.configuration_roberta import RobertaConfig


logger = logging.get_logger(__name__)

LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "luke-base": "https://huggingface.co/studio-ousia/luke-base/resolve/main/config.json",
    "luke-large": "https://huggingface.co/studio-ousia/luke-large/resolve/main/config.json",
}


class LukeConfig(RobertaConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.LukeModel`. It is used to
    instantiate a LUKE model according to the specified arguments, defining the model architecture. Configuration
    objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model outputs. Read the
    documentation from :class:`~transformers.PretrainedConfig` for more information. The
    :class:`~transformers.LukeConfig` class directly inherits :class:`~transformers.RobertaConfig`. It reuses the same
    defaults. Please check the parent class for more information.


    Args:
        entity_vocab_size (:obj:`int`, `optional`, defaults to 500000):
            Entity vocabulary size of the LUKE model. Defines the number of different entities that can be represented
            by the :obj:`entity_ids` passed when calling :class:`~transformers.LukeModel`.
        entity_emb_size (:obj:`int`, `optional`, defaults to 256):
            The number of dimensions of the entity embedding.
        use_entity_aware_attention (:obj:`bool`, defaults to :obj:`True`):
            Whether or not the model should use the entity-aware self-attention mechanism proposed in
            `LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention (Yamada et al.)
            <https://arxiv.org/abs/2010.01057>`__.

    Examples::
        >>> from transformers import LukeConfig, LukeModel
        >>> # Initializing a LUKE configuration
        >>> configuration = LukeConfig()
        >>> # Initializing a model from the configuration
        >>> model = LukeModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "luke"

    def __init__(
        self,
        vocab_size: int = 50267,
        entity_vocab_size: int = 500000,
        entity_emb_size: int = 256,
        use_entity_aware_attention=True,
        **kwargs
    ):
        """Constructs LukeConfig."""
        super(LukeConfig, self).__init__(vocab_size=vocab_size, **kwargs)

        self.entity_vocab_size = entity_vocab_size
        self.entity_emb_size = entity_emb_size
        self.use_entity_aware_attention = use_entity_aware_attention
