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

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...utils import (
    auto_docstring,
    logging,
)
from ..pp_lcnet_v3.modeling_pp_lcnet_v3 import make_divisible
from ..pp_ocrv5_server_rec.configuration_pp_ocrv5_server_rec import PPOCRV5ServerRecConfig
from ..pp_ocrv5_server_rec.modeling_pp_ocrv5_server_rec import (
    PPOCRV5ServerRecEncoderWithSVTR,
    PPOCRV5ServerRecForTextRecognition,
    PPOCRV5ServerRecModel,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PaddlePaddle/PP-OCRv5_mobile_rec_safetensors")
@strict
class PPOCRV5MobileRecConfig(PPOCRV5ServerRecConfig):
    def __post_init__(self, **kwargs):
        if self.conv_kernel_size is None:
            self.conv_kernel_size = [1, 3]
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="pp_lcnet_v3",
            default_config_kwargs={
                "scale": 0.75,
                "out_features": ["stage2", "stage3", "stage4", "stage5"],
                "out_indices": [2, 3, 4, 5],
                "divisor": 16,
            },
            **kwargs,
        )
        super().__post_init__(**kwargs)


class PPOCRV5MobileRecEncoderWithSVTR(PPOCRV5ServerRecEncoderWithSVTR):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        in_channels = make_divisible(  # noqa
            config.backbone_config.block_configs[-1][-1][2] * config.backbone_config.scale,
            config.backbone_config.divisor,
        )
        self.post_init()


class PPOCRV5MobileRecModel(PPOCRV5ServerRecModel):
    pass


@auto_docstring(custom_intro="PPOCRV5MobileRec model for text recognition tasks.")
class PPOCRV5MobileRecForTextRecognition(PPOCRV5ServerRecForTextRecognition):
    pass


__all__ = [
    "PPOCRV5MobileRecForTextRecognition",
    "PPOCRV5MobileRecConfig",
    "PPOCRV5MobileRecModel",
    "PPOCRV5MobileRecPreTrainedModel",  # noqa: F822
    "PPOCRV5MobileRecEncoderWithSVTR",
]
