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
""" CamemBERT configuration """


import logging

from .configuration_roberta import RobertaConfig


logger = logging.getLogger(__name__)

CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "camembert-base": "https://s3.amazonaws.com/models.huggingface.co/bert/camembert-base-config.json",
}


class CamembertConfig(RobertaConfig):
    r"""
        This is the configuration class to store the configuration of an :class:`~transformers.CamembertModel`.
        It is used to instantiate an Camembert model according to the specified arguments, defining the model
        architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
        the BERT `bert-base-uncased <https://huggingface.co/bert-base-uncased>`__ architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.

        The :class:`~transformers.CamembertConfig` class directly inherits :class:`~transformers.BertConfig`.
        It reuses the same defaults. Please check the parent class for more information.

        Example::

            from transformers import CamembertModel, CamembertConfig

            # Initializing a CamemBERT configuration
            configuration = CamembertConfig()

            # Initializing a model from the configuration
            model = CamembertModel(configuration)

            # Accessing the model configuration
            configuration = model.config

        Attributes:
            pretrained_config_archive_map (Dict[str, str]):
                A dictionary containing all the available pre-trained checkpoints.
    """
    pretrained_config_archive_map = CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "camembert"
