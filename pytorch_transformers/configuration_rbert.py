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
""" BERT model configuration """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from pytorch_transformers import BertConfig
from .configuration_roberta import RobertaConfig

logger = logging.getLogger(__name__)


class RBertConfig(BertConfig):
    r"""
        :class:`~pytorch_transformers.RBertConfig` is a small extension of the `~pytorch_transformers.BertConfig`,
        required for the :class:`~pytorch_transformers.BertForRelationshipClassification` model

        RBERT uses single character tokens to delimit entities in a sentence.

        config = RBertConfig.from_pretrained("config_name",
                                          num_labels=num_labels, finetuning_task=args.task_name,
                                          entity_1_token_id=tokenizer.entity_1_token_id,
                                          entity_2_token_id=tokenizer.entity_2_token_id,
                                          )


        Arguments:
            entity_1_token_id: the token ID of the first entity delimiter
            entity_2_token_id: the token ID if the second entity delimiter
    """
    def __init__(self,
                 entity_1_token_id=1001,
                 entity_2_token_id=1002,
                 **kwargs):
        super(RBertConfig, self).__init__(**kwargs)
        self.entity_2_token_id = entity_2_token_id
        self.entity_1_token_id = entity_1_token_id


class RBertForRobertaConfig(RobertaConfig):
    r"""
        :class:`~pytorch_transformers.RBertForRobertaConfig` is a small extension of the :class
        `~pytorch_transformers.RobertaConfig`, required for the :class:`~pytorch_transformers.RobertaForRelationshipClassification` model

        RBERT uses single character tokens to delimit entities in a sentence.

        config = RBertForRobertaConfig.from_pretrained("config_name",
                                          num_labels=num_labels, finetuning_task=args.task_name,
                                          entity_1_token_id=tokenizer.entity_1_token_id,
                                          entity_2_token_id=tokenizer.entity_2_token_id,
                                          )


        Arguments:
            entity_1_token_id: the token ID of the first entity delimiter
            entity_2_token_id: the token ID if the second entity delimiter
    """
    def __init__(self,
                 entity_1_token_id=1001,
                 entity_2_token_id=1002,
                 **kwargs):
        super(RBertForRobertaConfig, self).__init__(**kwargs)
        self.entity_2_token_id = entity_2_token_id
        self.entity_1_token_id = entity_1_token_id
