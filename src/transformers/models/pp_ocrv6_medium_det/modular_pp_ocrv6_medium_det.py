# Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
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

from huggingface_hub.dataclasses import strict

from ...backbone_utils import (
    consolidate_backbone_kwargs_to_config,
)
from ...configuration_utils import PreTrainedConfig
from ...utils import (
    auto_docstring,
    logging,
)
from ..pp_ocrv5_mobile_det.modeling_pp_ocrv5_mobile_det import (
    PPOCRV5MobileDetHead,
)
from ..pp_ocrv5_server_det.configuration_pp_ocrv5_server_det import PPOCRV5ServerDetConfig
from ..pp_ocrv5_server_det.modeling_pp_ocrv5_server_det import (
    PPOCRV5ServerDetForObjectDetection,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PaddlePaddle/PP-OCRv6_medium_det_safetensors")
@strict
class PPOCRV6MediumDetConfig(PPOCRV5ServerDetConfig):
    hidden_act = AttributeError()

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="pp_lcnet_v4",
            **kwargs,
        )
        PreTrainedConfig.__post_init__(**kwargs)


class PPOCRV6MediumDetHead(PPOCRV5MobileDetHead):
    pass


@auto_docstring(custom_intro="PPOCRV6MediumDet model for text detection tasks.")
class PPOCRV6MediumDetForObjectDetection(PPOCRV5ServerDetForObjectDetection):
    pass


__all__ = [
    "PPOCRV6MediumDetForObjectDetection",
    "PPOCRV6MediumDetConfig",
    "PPOCRV6MediumDetModel",  # noqa: F822
    "PPOCRV6MediumDetPreTrainedModel",  # noqa: F822
]
