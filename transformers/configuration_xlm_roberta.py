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
""" XLM-RoBERTa configuration """

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

from .configuration_roberta import RobertaConfig

logger = logging.getLogger(__name__)

XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'xlm-roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-base-config.json",
    'xlm-roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-config.json",
    'xlm-roberta-large-finetuned-conll02-dutch': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-dutch-config.json",
    'xlm-roberta-large-finetuned-conll02-spanish': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-spanish-config.json",
    'xlm-roberta-large-finetuned-conll03-english': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-english-config.json",
    'xlm-roberta-large-finetuned-conll03-german': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-german-config.json",
}


class XLMRobertaConfig(RobertaConfig):
    pretrained_config_archive_map = XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP
