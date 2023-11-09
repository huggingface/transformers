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
    model_type = "textnet"

    def __init__(
        self,
        kernel_size=3,
        stride=2,
        dilation=1,
        groups=1,
        bias=False,
        has_shuffle=False,
        in_channels=3,
        out_channels=64,
        act_func="relu",
        stage1_in_channels=[64, 64, 64],
        stage1_out_channels=[64, 64, 64],
        stage1_kernel_size=[[3, 3], [3, 3], [3, 3]],
        stage1_stride=[1, 2, 1],
        stage1_dilation=[1, 1, 1],
        stage1_groups=[1, 1, 1],
        stage2_in_channels=[64, 128, 128, 128],
        stage2_out_channels=[128, 128, 128, 128],
        stage2_kernel_size=[[3, 3], [1, 3], [3, 3], [3, 1]],
        stage2_stride=[2, 1, 1, 1],
        stage2_dilation=[1, 1, 1, 1],
        stage2_groups=[1, 1, 1, 1],
        stage3_in_channels=[128, 256, 256, 256],
        stage3_out_channels=[256, 256, 256, 256],
        stage3_kernel_size=[[3, 3], [3, 3], [3, 1], [1, 3]],
        stage3_stride=[2, 1, 1, 1],
        stage3_dilation=[1, 1, 1, 1],
        stage3_groups=[1, 1, 1, 1],
        stage4_in_channels=[256, 512, 512, 512],
        stage4_out_channels=[512, 512, 512, 512],
        stage4_kernel_size=[[3, 3], [3, 1], [1, 3], [3, 3]],
        stage4_stride=[2, 1, 1, 1],
        stage4_dilation=[1, 1, 1, 1],
        stage4_groups=[1, 1, 1, 1],
        hidden_sizes=[64, 64, 128, 256, 512],
        initializer_range=0.02,
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_func = act_func

        self.stage1_in_channels = stage1_in_channels
        self.stage1_out_channels = stage1_out_channels
        self.stage1_kernel_size = stage1_kernel_size
        self.stage1_stride = stage1_stride
        self.stage1_dilation = stage1_dilation
        self.stage1_groups = stage1_groups

        self.stage2_in_channels = stage2_in_channels
        self.stage2_out_channels = stage2_out_channels
        self.stage2_kernel_size = stage2_kernel_size
        self.stage2_stride = stage2_stride
        self.stage2_dilation = stage2_dilation
        self.stage2_groups = stage2_groups

        self.stage3_in_channels = stage3_in_channels
        self.stage3_out_channels = stage3_out_channels
        self.stage3_kernel_size = stage3_kernel_size
        self.stage3_stride = stage3_stride
        self.stage3_dilation = stage3_dilation
        self.stage3_groups = stage3_groups

        self.stage4_in_channels = stage4_in_channels
        self.stage4_out_channels = stage4_out_channels
        self.stage4_kernel_size = stage4_kernel_size
        self.stage4_stride = stage4_stride
        self.stage4_dilation = stage4_dilation
        self.stage4_groups = stage4_groups

        self.initializer_range = initializer_range
        self.hidden_sizes = hidden_sizes

        self.depths = [
            len(self.stage1_out_channels),
            len(self.stage2_out_channels),
            len(self.stage3_out_channels),
            len(self.stage4_out_channels),
        ]
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, 5)]
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
