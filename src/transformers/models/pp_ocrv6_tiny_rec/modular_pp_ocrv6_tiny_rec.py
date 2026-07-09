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

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...backbone_utils import (
    consolidate_backbone_kwargs_to_config,
)
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ..pp_ocrv6_small_rec.configuration_pp_ocrv6_small_rec import PPOCRV6SmallRecConfig
from ..pp_ocrv6_small_rec.modeling_pp_ocrv6_small_rec import (
    PPOCRV6SmallRecForTextRecognition,
    PPOCRV6SmallRecPreTrainedModel,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PaddlePaddle/PP-OCRv6_tiny_rec_safetensors")
@strict
class PPOCRV6TinyRecConfig(PPOCRV6SmallRecConfig):
    head_out_channels: int = 6625
    conv_kernel_size = AttributeError()
    mlp_ratio = AttributeError()
    depth = AttributeError()
    hidden_act = AttributeError()
    attention_dropout = AttributeError()
    layer_norm_eps = AttributeError()
    num_attention_heads = AttributeError()
    qkv_bias = AttributeError()

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="pp_lcnet_v4",
            **kwargs,
        )
        PreTrainedConfig.__post_init__(**kwargs)


@auto_docstring
class PPOCRV6TinyRecPreTrainedModel(PPOCRV6SmallRecPreTrainedModel):
    supports_gradient_checkpointing = False
    _no_split_modules = []
    _can_record_outputs = {}


class PPOCRV6TinyRecHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.backbone_config.block_configs[-1][-1][2]
        mid_channels = config.hidden_size

        self.conv1 = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=5,
            padding=2,
            groups=in_channels,
            bias=False,
        )
        self.norm1 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm1d(in_channels)
        self.act_fn = nn.Hardswish()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, config.head_out_channels)

    def forward(self, hidden_states: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]):
        hidden_states = hidden_states.squeeze(2)
        hidden_states = self.act_fn(self.norm1(self.conv1(hidden_states)))
        hidden_states = self.act_fn(self.norm2(self.conv2(hidden_states)))

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.fc2(hidden_states)

        hidden_states = F.softmax(hidden_states, dim=2, dtype=torch.float32).to(hidden_states.dtype)

        return hidden_states


@auto_docstring(custom_intro="PPOCR6TinyRec model for text recognition tasks.")
class PPOCRV6TinyRecForTextRecognition(PPOCRV6SmallRecForTextRecognition):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithNoAttention:
        outputs = self.model(pixel_values, **kwargs)
        head_outputs = self.head(outputs.last_hidden_state, **kwargs)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=head_outputs,
            hidden_states=outputs.hidden_states,
        )


__all__ = [
    "PPOCRV6TinyRecForTextRecognition",
    "PPOCRV6TinyRecConfig",
    "PPOCRV6TinyRecModel",  # noqa: F822
    "PPOCRV6TinyRecPreTrainedModel",
]
