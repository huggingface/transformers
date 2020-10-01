#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
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

import torch

from .configuration_blenderbot import BlenderbotConfig
from .file_utils import add_start_docstrings
from .modeling_bart import BartForConditionalGeneration


BLENDER_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matters related to general usage and behavior.

    Parameters:
        config (:class:`~transformers.BlenderbotConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.

"""

BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST = ["facebook/blenderbot-3B", "facebook/blenderbot-90M"]


@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.", BLENDER_START_DOCSTRING
)
class BlenderbotForConditionalGeneration(BartForConditionalGeneration):
    """
    This class overrides :class:`~transformers.BartForConditionalGeneration`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = BlenderbotConfig

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        logits[:, self.config.bos_token_id] = -torch.finfo(torch.float16).max  # near infinity fp16
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits
