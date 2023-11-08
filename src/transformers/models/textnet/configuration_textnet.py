# coding=utf-8
# Copyright The HuggingFace Inc. team. All rights reserved.
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
""" TextNet model configuration"""
from transformers import PretrainedConfig
from transformers.utils import logging
from transformers.utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices


logger = logging.get_logger(__name__)

TEXTNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "fast_base_tt_800_finetune_ic17mlt": (
        "https://huggingface.co/Raghavan/fast_base_tt_800_finetune_ic17mlt/raw/main/config.json"
    ),
}


class TextNetConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    [Raghavan/fast_base_tt_800_finetune_ic17mlt](https://huggingface.co/Raghavan/fast_base_tt_800_finetune_ic17mlt)
    """

    def __init__(
        self,
        backbone_kernel_size=3,
        backbone_stride=2,
        backbone_dilation=1,
        backbone_groups=1,
        backbone_bias=False,
        backbone_has_shuffle=False,
        backbone_in_channels=3,
        backbone_out_channels=64,
        backbone_use_bn=True,
        backbone_act_func="relu",
        backbone_dropout_rate=0,
        backbone_ops_order="weight_bn_act",
        backbone_stage1_in_channels=[64, 64, 64],
        backbone_stage1_out_channels=[64, 64, 64],
        backbone_stage1_kernel_size=[[3, 3], [3, 3], [3, 3]],
        backbone_stage1_stride=[1, 2, 1],
        backbone_stage1_dilation=[1, 1, 1],
        backbone_stage1_groups=[1, 1, 1],
        backbone_stage2_in_channels=[64, 128, 128, 128],
        backbone_stage2_out_channels=[128, 128, 128, 128],
        backbone_stage2_kernel_size=[[3, 3], [1, 3], [3, 3], [3, 1]],
        backbone_stage2_stride=[2, 1, 1, 1],
        backbone_stage2_dilation=[1, 1, 1, 1],
        backbone_stage2_groups=[1, 1, 1, 1],
        backbone_stage3_in_channels=[128, 256, 256, 256],
        backbone_stage3_out_channels=[256, 256, 256, 256],
        backbone_stage3_kernel_size=[[3, 3], [3, 3], [3, 1], [1, 3]],
        backbone_stage3_stride=[2, 1, 1, 1],
        backbone_stage3_dilation=[1, 1, 1, 1],
        backbone_stage3_groups=[1, 1, 1, 1],
        backbone_stage4_in_channels=[256, 512, 512, 512],
        backbone_stage4_out_channels=[512, 512, 512, 512],
        backbone_stage4_kernel_size=[[3, 3], [3, 1], [1, 3], [3, 3]],
        backbone_stage4_stride=[2, 1, 1, 1],
        backbone_stage4_dilation=[1, 1, 1, 1],
        backbone_stage4_groups=[1, 1, 1, 1],
        hidden_sizes=[64, 64, 128, 256, 512],
        initializer_range=0.02,
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.backbone_kernel_size = backbone_kernel_size
        self.backbone_stride = backbone_stride
        self.backbone_dilation = backbone_dilation
        self.backbone_groups = backbone_groups
        self.backbone_bias = backbone_bias
        self.backbone_has_shuffle = backbone_has_shuffle
        self.backbone_in_channels = backbone_in_channels
        self.backbone_out_channels = backbone_out_channels
        self.backbone_use_bn = backbone_use_bn
        self.backbone_act_func = backbone_act_func
        self.backbone_dropout_rate = backbone_dropout_rate
        self.backbone_ops_order = backbone_ops_order

        self.backbone_stage1_in_channels = backbone_stage1_in_channels
        self.backbone_stage1_out_channels = backbone_stage1_out_channels
        self.backbone_stage1_kernel_size = backbone_stage1_kernel_size
        self.backbone_stage1_stride = backbone_stage1_stride
        self.backbone_stage1_dilation = backbone_stage1_dilation
        self.backbone_stage1_groups = backbone_stage1_groups

        self.backbone_stage2_in_channels = backbone_stage2_in_channels
        self.backbone_stage2_out_channels = backbone_stage2_out_channels
        self.backbone_stage2_kernel_size = backbone_stage2_kernel_size
        self.backbone_stage2_stride = backbone_stage2_stride
        self.backbone_stage2_dilation = backbone_stage2_dilation
        self.backbone_stage2_groups = backbone_stage2_groups

        self.backbone_stage3_in_channels = backbone_stage3_in_channels
        self.backbone_stage3_out_channels = backbone_stage3_out_channels
        self.backbone_stage3_kernel_size = backbone_stage3_kernel_size
        self.backbone_stage3_stride = backbone_stage3_stride
        self.backbone_stage3_dilation = backbone_stage3_dilation
        self.backbone_stage3_groups = backbone_stage3_groups

        self.backbone_stage4_in_channels = backbone_stage4_in_channels
        self.backbone_stage4_out_channels = backbone_stage4_out_channels
        self.backbone_stage4_kernel_size = backbone_stage4_kernel_size
        self.backbone_stage4_stride = backbone_stage4_stride
        self.backbone_stage4_dilation = backbone_stage4_dilation
        self.backbone_stage4_groups = backbone_stage4_groups

        self.initializer_range = initializer_range
        self.hidden_sizes = hidden_sizes

        self.depths = [
            len(self.backbone_stage1_out_channels),
            len(self.backbone_stage2_out_channels),
            len(self.backbone_stage3_out_channels),
            len(self.backbone_stage4_out_channels),
        ]
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, 5)]
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
