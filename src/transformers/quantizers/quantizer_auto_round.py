# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import importlib
from typing import TYPE_CHECKING

from packaging import version

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_auto_round_available, is_torch_available, logging


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class AutoRoundQuantizer(HfQuantizer):
    """
    Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs (https://arxiv.org/abs/2309.05516)
    """

    # AutoRound requires data calibration - we support only inference
    requires_calibration = True

    required_packages = ["auto-round", "accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, device_map, **kwargs):
        if not is_auto_round_available():
            raise ImportError(
                "Loading an AUTOROUND quantized model requires auto-round library (`pip install auto-round`)"
            )
        elif version.parse(importlib.metadata.version("auto_round")) <= version.parse("0.2.0"):
            raise ImportError(
                "You need a version of auto_round > 0.2.0 to use AutoRound: `pip install --upgrade auto-round`"
            )

    def update_torch_dtype(self, torch_dtype):
        if torch_dtype is None:
            torch_dtype = torch.float16
        elif torch_dtype != torch.float16:
            logger.warning("We suggest you to set `torch_dtype=torch.float16` for better efficiency with AutoRound.")
        return torch_dtype

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        from ..integrations import convert_auto_round_model

        model = convert_auto_round_model(model, self.quantization_config)

    def _process_model_after_weight_loading(self, model):
        from ..integrations import post_init_auto_round_model

        model = post_init_auto_round_model(model)

    @property
    def is_serializable(self):
        return True

    @property
    def is_trainable(self):
        return True
