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
""" XLM-RoBERTa configuration"""
from collections import OrderedDict
from typing import Mapping

from transformers import XLMRobertaConfig

from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

UNITE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "unite-up": "https://huggingface.co/ywan/unite-up/tree/main/config.json",
    "unite-mup": "https://huggingface.co/ywan/unite-mup/tree/main/config.json",
}


class UniTEConfig(XLMRobertaConfig):
    """
    This class overrides [`RobertaConfig`]. Please check the superclass for the appropriate documentation alongside
    usage examples.
    """

    model_type = "unite"

