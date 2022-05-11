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

from ...onnx import OnnxConfig
from ...utils import logging
from ..roberta.configuration_roberta import RobertaConfig


logger = logging.get_logger(__name__)

XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "xlm-roberta-base": "https://huggingface.co/xlm-roberta-base/resolve/main/config.json",
    "xlm-roberta-large": "https://huggingface.co/xlm-roberta-large/resolve/main/config.json",
    "xlm-roberta-large-finetuned-conll02-dutch": "https://huggingface.co/xlm-roberta-large-finetuned-conll02-dutch/resolve/main/config.json",
    "xlm-roberta-large-finetuned-conll02-spanish": "https://huggingface.co/xlm-roberta-large-finetuned-conll02-spanish/resolve/main/config.json",
    "xlm-roberta-large-finetuned-conll03-english": "https://huggingface.co/xlm-roberta-large-finetuned-conll03-english/resolve/main/config.json",
    "xlm-roberta-large-finetuned-conll03-german": "https://huggingface.co/xlm-roberta-large-finetuned-conll03-german/resolve/main/config.json",
}


class XLMRobertaConfig(RobertaConfig):
    """
    This class overrides [`RobertaConfig`]. Please check the superclass for the appropriate documentation alongside
    usage examples. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    XLMRoBERTa [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) architecture.
    """

    model_type = "xlm-roberta"


# Copied from transformers.models.roberta.configuration_roberta.RobertaOnnxConfig with Roberta->XLMRoberta
class XLMRobertaOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )
