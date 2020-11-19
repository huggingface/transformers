# coding=utf-8
# Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
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
""" XLM-ProphetNet model configuration """


from .configuration_prophetnet import ProphetNetConfig
from .utils import logging


logger = logging.get_logger(__name__)

XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/xprophetnet-large-wiki100-cased": "https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased/resolve/main/config.json",
}


class XLMProphetNetConfig(ProphetNetConfig):
    """
    This class overrides :class:`~transformers.ProphetNetConfig`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    model_type = "xlm-prophetnet"
