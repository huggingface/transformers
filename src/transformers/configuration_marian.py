# coding=utf-8
# Copyright 2020 The OPUS-NMT Team, Marian team, and The HuggingFace Inc. team.
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
""" Marian model configuration """

from .configuration_bart import BartConfig


PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "marian-en-de": "https://s3.amazonaws.com/models.huggingface.co/bert/Helsinki-NLP/opus-mt-en-de/config.json",
}


class MarianConfig(BartConfig):
    pretrained_config_archive_map = PRETRAINED_CONFIG_ARCHIVE_MAP
