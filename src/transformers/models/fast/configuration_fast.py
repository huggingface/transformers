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
""" Fast model configuration"""
from transformers import CONFIG_MAPPING, PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

FAST_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "fast_base_tt_800_finetune_ic17mlt": (
        "https://huggingface.co/Raghavan/fast_base_tt_800_finetune_ic17mlt/raw/main/config.json"
    ),
}


class FastConfig(PretrainedConfig):
    r"""
    [Raghavan/fast_base_tt_800_finetune_ic17mlt](https://huggingface.co/Raghavan/fast_base_tt_800_finetune_ic17mlt)
    """

    def __init__(
        self,
        use_timm_backbone=True,
        backbone_config=None,
        num_channels=3,
        neck_in_channels=[64, 128, 256, 512],
        neck_out_channels=[128, 128, 128, 128],
        neck_kernel_size=[[3, 3], [3, 3], [3, 3], [3, 3]],
        neck_stride=[1, 1, 1, 1],
        neck_dilation=[1, 1, 1, 1],
        neck_groups=[1, 1, 1, 1],
        head_pooling_size=9,
        head_dropout_ratio=0,
        head_conv_in_channels=512,
        head_conv_out_channels=128,
        head_conv_kernel_size=[3, 3],
        head_conv_stride=1,
        head_conv_dilation=1,
        head_conv_groups=1,
        head_final_kernel_size=1,
        head_final_stride=1,
        head_final_dilation=1,
        head_final_groups=1,
        head_final_bias=False,
        head_final_has_shuffle=False,
        head_final_in_channels=128,
        head_final_out_channels=5,
        head_final_use_bn=False,
        head_final_act_func=None,
        head_final_dropout_rate=0,
        head_final_ops_order="weight",
        loss_bg=False,
        backbone="resnet50",
        use_pretrained_backbone=True,
        dilation=False,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if backbone_config is not None and use_timm_backbone:
            raise ValueError("You can't specify both `backbone_config` and `use_timm_backbone`.")

        if not use_timm_backbone:
            if backbone_config is None:
                logger.info(
                    "`backbone_config` is `None`. Initializing the config with the default `TextNet` backbone."
                )
                backbone_config = CONFIG_MAPPING["textnet"](out_features=["stage1", "stage2", "stage3", "stage4"])
            elif isinstance(backbone_config, dict):
                backbone_model_type = backbone_config.get("model_type")
                config_class = CONFIG_MAPPING[backbone_model_type]
                backbone_config = config_class.from_dict(backbone_config)
            # set timm attributes to None
            dilation, backbone, use_pretrained_backbone = None, None, None

        self.use_timm_backbone = use_timm_backbone
        self.backbone_config = backbone_config
        self.num_channels = num_channels

        self.neck_in_channels = neck_in_channels
        self.neck_out_channels = neck_out_channels
        self.neck_kernel_size = neck_kernel_size
        self.neck_stride = neck_stride
        self.neck_dilation = neck_dilation
        self.neck_groups = neck_groups

        self.head_pooling_size = head_pooling_size
        self.head_dropout_ratio = head_dropout_ratio

        self.head_conv_in_channels = head_conv_in_channels
        self.head_conv_out_channels = head_conv_out_channels
        self.head_conv_kernel_size = head_conv_kernel_size
        self.head_conv_stride = head_conv_stride
        self.head_conv_dilation = head_conv_dilation
        self.head_conv_groups = head_conv_groups

        self.head_final_kernel_size = head_final_kernel_size
        self.head_final_stride = head_final_stride
        self.head_final_dilation = head_final_dilation
        self.head_final_groups = head_final_groups
        self.head_final_bias = head_final_bias
        self.head_final_has_shuffle = head_final_has_shuffle
        self.head_final_in_channels = head_final_in_channels
        self.head_final_out_channels = head_final_out_channels
        self.head_final_use_bn = head_final_use_bn
        self.head_final_act_func = head_final_act_func
        self.head_final_dropout_rate = head_final_dropout_rate
        self.head_final_ops_order = head_final_ops_order

        self.loss_bg = loss_bg
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.dilation = dilation

        self.initializer_range = initializer_range

    @classmethod
    def from_backbone_config(cls, backbone_config: PretrainedConfig, **kwargs):
        """Instantiate a [`FastConfig`] (or a derived class) from a pre-trained backbone model configuration.

        Args:
            backbone_config ([`PretrainedConfig`]):
                The backbone configuration.
        Returns:
            [`DetrConfig`]: An instance of a configuration object
        """
        return cls(backbone_config=backbone_config, **kwargs)
