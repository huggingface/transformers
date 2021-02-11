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
""" I-BERT configuration """

from ...utils import logging
from ..roberta.configuration_roberta import RobertaConfig


logger = logging.get_logger(__name__)

IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "kssteven/ibert-roberta-base": "https://huggingface.co/roberta-base/resolve/main/config.json",
    "kssteven/ibert-roberta-large": "https://huggingface.co/roberta-large/resolve/main/config.json",
    "kssteven/ibert-roberta-large-mnli": "https://huggingface.co/roberta-large-mnli/resolve/main/config.json",
}


class IBertConfig(RobertaConfig):
    """
    This class overrides :class:`~transformers.RobertaConfig`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    model_type = "ibert"
