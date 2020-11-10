# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" RoBERTa configuration """

from .configuration_bert import BertConfig
from .utils import logging


logger = logging.get_logger(__name__)

ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "roberta-base": "https://huggingface.co/roberta-base/resolve/main/config.json",
    "roberta-large": "https://huggingface.co/roberta-large/resolve/main/config.json",
    "roberta-large-mnli": "https://huggingface.co/roberta-large-mnli/resolve/main/config.json",
    "distilroberta-base": "https://huggingface.co/distilroberta-base/resolve/main/config.json",
    "roberta-base-openai-detector": "https://huggingface.co/roberta-base-openai-detector/resolve/main/config.json",
    "roberta-large-openai-detector": "https://huggingface.co/roberta-large-openai-detector/resolve/main/config.json",
}


class RobertaConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.RobertaModel` or a
    :class:`~transformers.TFRobertaModel`. It is used to instantiate a RoBERTa model according to the specified
    arguments, defining the model architecture.


    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    The :class:`~transformers.RobertaConfig` class directly inherits :class:`~transformers.BertConfig`. It reuses the
    same defaults. Please check the parent class for more information.

    Examples::

        >>> from transformers import RobertaConfig, RobertaModel

        >>> # Initializing a RoBERTa configuration
        >>> configuration = RobertaConfig()

        >>> # Initializing a model from the configuration
        >>> model = RobertaModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "roberta"

    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
