# coding=utf-8
# Copyright Facebook and The HuggingFace Inc. team. All rights reserved.
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
""" ESM model configuration """

from ...utils import logging
from ..bert.configuration_bert import BertConfig

logger = logging.get_logger(__name__)

ESM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/esm1b": "https://huggingface.co/facebook/esm1b/resolve/main/config.json",
    # See all ESM models at https://huggingface.co/models?filter=esm
}


class ESMConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.ESMModel`. It is used to instantiate a ESM model according to the specified
    arguments, defining the model architecture.


    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    The :class:`~transformers.ESMConfig` class directly inherits :class:`~transformers.BertConfig`. It reuses the
    same defaults. Please check the parent class for more information.

    Examples::

        >>> from transformers import ESMConfig, ESMModel

        >>> # Initializing a ESM configuration
        >>> configuration = ESMConfig()

        >>> # Initializing a model from the configuration
        >>> model = ESMModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "esm"

    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, encoder_keep_prob=0.88, **kwargs):
        """Constructs ESMConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.encoder_keep_prob = encoder_keep_prob
