# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import torch

from .utils import logging


logger = logging.get_logger(__name__)


def detect_overflow(var, ctx):
    """
    Report the count of ``nan`` and ``inf`` entries in the tensor.

    This is useful for detecting overflows/underflows and best to call right after the function that did some math that
    modified the variable in question.

    Args:
        var: tensor variable to check
        ctx: the message to print as a context
    """
    if torch.isnan(var).any().item():
        logger.warning(f"{ctx} has nans")
    if torch.isinf(var).any().item():
        logger.warning(f"{ctx} has inf")

    # if needed to monitor large elements can enable the following
    if 0:
        n100 = var[torch.ge(var.abs(), 100)]
        if n100.numel() > 0:
            logger.warning(f"{ctx}:  n100={n100.numel()}")
        n1000 = var[torch.ge(var.abs(), 1000)]
        if n1000.numel() > 0:
            logger.warning(f"{ctx}: n1000={n1000.numel()}")
