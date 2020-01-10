# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Musixmatch spa and The HuggingFace Inc. team.
# This code is referring to the Camembert code, just to simplify not for copying
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

""" UmBERTo configuration """
# This code is referring to the Camembert code, just to simplify


import logging

from .configuration_roberta import RobertaConfig


logger = logging.getLogger(__name__)

UMBERTO_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "umberto-commoncrawl-cased-v1": "https://mxmdownloads.s3.amazonaws.com/umberto/umberto-commoncrawl-cased-v1-config.json",
    "umberto-wikipedia-uncased-v1": "https://mxmdownloads.s3.amazonaws.com/umberto/umberto-wikipedia-uncased-v1-config.json",
}


class UmbertoConfig(RobertaConfig):
    pretrained_config_archive_map = UMBERTO_PRETRAINED_CONFIG_ARCHIVE_MAP
