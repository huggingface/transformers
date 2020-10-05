#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Facebook, Inc. and Huggingface, 2020
#
# This source code is licensed under the MIT license found in the;
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
# LICENSE file in the root directory of this source tree.
from .configuration_bart import BartConfig


BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/blenderbot-3B": "https://cdn.huggingface.co/facebook/blenderbot-3B/config.json",
    "facebook/blenderbot-90M": "https://cdn.huggingface.co/facebook/blenderbot-90M/config.json",
}


class BlenderbotConfig(BartConfig):
    """
    This class overrides :class:`~transformers.BartConfig`. Please check the superclass for the appropriate documentation alongside usage examples.
    To see the default values, run the Usage example below. For a deeper understanding of how the config affects layernorm placement, see https://tinyurl.com/y66r9gnh

    Usage:

        >>> from transformers import BlenderbotConfig
        >>> config_90 = BlenderbotConfig.from_pretrained("facebook/blenderbot-90M")
        >>> config_90.to_diff_dict()  # show interesting Values.
        >>> configuration_3B = BlenderbotConfig("facebook/blenderbot-3B")
        >>> configuration_3B.to_diff_dict()

    """

    model_type = "blenderbot"
