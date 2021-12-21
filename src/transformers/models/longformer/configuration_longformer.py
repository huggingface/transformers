# coding=utf-8
# Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
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
""" Longformer configuration """
from collections import OrderedDict
from typing import List, Mapping, Union

from ...onnx import OnnxConfig
from ...utils import logging
from ..roberta.configuration_roberta import RobertaConfig


logger = logging.get_logger(__name__)

LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "allenai/longformer-base-4096": "https://huggingface.co/allenai/longformer-base-4096/resolve/main/config.json",
    "allenai/longformer-large-4096": "https://huggingface.co/allenai/longformer-large-4096/resolve/main/config.json",
    "allenai/longformer-large-4096-finetuned-triviaqa": "https://huggingface.co/allenai/longformer-large-4096-finetuned-triviaqa/resolve/main/config.json",
    "allenai/longformer-base-4096-extra.pos.embd.only": "https://huggingface.co/allenai/longformer-base-4096-extra.pos.embd.only/resolve/main/config.json",
    "allenai/longformer-large-4096-extra.pos.embd.only": "https://huggingface.co/allenai/longformer-large-4096-extra.pos.embd.only/resolve/main/config.json",
}


class LongformerConfig(RobertaConfig):
    r"""
    This is the configuration class to store the configuration of a [`LongformerModel`] or a
    [`TFLongformerModel`]. It is used to instantiate a Longformer model according to the specified
    arguments, defining the model architecture.

    This is the configuration class to store the configuration of a [`LongformerModel`]. It is used
    to instantiate an Longformer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the RoBERTa
    [roberta-base](https://huggingface.co/roberta-base) architecture with a sequence length 4,096.

    The [`LongformerConfig`] class directly inherits [`RobertaConfig`]. It reuses
    the same defaults. Please check the parent class for more information.

    Args:
        attention_window (`int` or `List[int]`, *optional*, defaults to 512):
            Size of an attention window around each token. If an `int`, use the same size for all layers. To
            specify a different window size for each layer, use a `List[int]` where `len(attention_window) == num_hidden_layers`.

    Example:

    ```python
    >>> from transformers import LongformerConfig, LongformerModel

    >>> # Initializing a Longformer configuration
    >>> configuration = LongformerConfig()

    >>> # Initializing a model from the configuration
    >>> model = LongformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "longformer"

    def __init__(self, attention_window: Union[List[int], int] = 512, sep_token_id: int = 2, **kwargs):
        super().__init__(sep_token_id=sep_token_id, **kwargs)
        self.attention_window = attention_window


class LongformerOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
            ]
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([("last_hidden_state", {0: "batch", 1: "sequence"}), ("pooler_output", {0: "batch"})])
