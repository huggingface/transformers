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
""" data2vec configuration"""
from collections import OrderedDict
from typing import Mapping

from ...onnx import OnnxConfig
from ...utils import logging
from ..bert.configuration_bert import BertConfig


logger = logging.get_logger(__name__)

DATA2VEC_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "data2vec": "https://huggingface.co/data2vec/resolve/main/config.json",
}


class Data2VecConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a [`Data2VecModel`] or a [`TFData2VecModel`]. It is
    used to instantiate a data2vec model according to the specified arguments, defining the model architecture.


    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    The [`Data2VecConfig`] class directly inherits [`BertConfig`]. It reuses the same defaults. Please check the parent
    class for more information.

    Examples:

    ```python
    >>> from transformers import Data2VecConfig, Data2VecModel

    >>> # Initializing a data2vec configuration
    >>> configuration = Data2VecConfig()

    >>> # Initializing a model from the configuration
    >>> model = Data2VecModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "data2vec"

    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs):
        """Constructs Data2VecConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class Data2VecOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
            ]
        )
