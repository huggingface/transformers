# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
    "textnet-base": ("https://huggingface.co/Raghavan/textnet-base/blob/main/config.json"),
}


class TextNetConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastForSceneTextRecognition`]. It is used to
    instantiate a FastForSceneTextRecognition model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastForSceneTextRecognition.
    [Raghavan/fast_base_tt_800_finetune_ic17mlt](https://huggingface.co/Raghavan/fast_base_tt_800_finetune_ic17mlt)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        kernel_size (`int`, *optional*, defaults to 3):
            The kernel size for the initial convolution layer.
        stride (`int`, *optional*, defaults to 2):
            The stride for the initial convolution layer.
        in_channels (`int`, *optional*, defaults to 3):
            The num of channels in input for the initial convolution layer.
        out_channels (`int`, *optional*, defaults to 64):
            The num of channels in out for the initial convolution layer.
        act_func (`str`, *optional*, defaults to `"relu"`):
            The activation function for the initial convolution layer.
        stage1_in_channels (`List[int]`, *optional*, defaults to `[64, 64, 64]`):
            The num of channels in input for list of conv in stage 1.
        stage1_out_channels (`List[int]`, *optional*, defaults to `[64, 64, 64]`):
            The num of channels in output for list of conv in stage 1.Should be of same length os `stage1_in_channels`
        stage1_kernel_size (`List[int]`, *optional*, defaults to `[[3, 3], [3, 3], [3, 3]]`):
            The kernel sizes for list of conv in stage 1.Should be of same length os `stage1_in_channels`
        stage1_stride (`List[int]`, *optional*, defaults to `[1, 2, 1]`):
            The strides for list of conv in stage 1.Should be of same length os `stage1_in_channels`
        stage2_in_channels (`List[int]`, *optional*, defaults to `[64, 128, 128, 128]`):
            The num of channels in input for list of conv in stage 2.
        stage2_out_channels (`List[int]`, *optional*, defaults to `[128, 128, 128, 128]`):
            The num of channels in output for list of conv in stage 2.Should be of same length os `stage2_in_channels`
        stage2_kernel_size (`List[List[int]]`, *optional*, defaults to `[[3, 3], [1, 3], [3, 3], [3, 1]]`):
            The kernel sizes for list of conv in stage 2.Should be of same length os `stage2_in_channels`
        stage2_stride (`List[int]`, *optional*, defaults to `[2, 1, 1, 1]`):
            The strides for list of conv in stage 2.Should be of same length os `stage2_in_channels`
        stage3_in_channels (`List[int]`, *optional*, defaults to `[128, 256, 256, 256]`):
            The num of channels in input for list of conv in stage 3.
        stage3_out_channels (`List[int]`, *optional*, defaults to `[256, 256, 256, 256]`):
            The num of channels in output for list of conv in stage 3.Should be of same length os `stage3_in_channels`
        stage3_kernel_size (`List[List[int]]`, *optional*, defaults to `[[3, 3], [3, 3], [3, 1], [1, 3]]`):
            The kernel sizes for list of conv in stage 3.Should be of same length os `stage3_in_channels`
        stage3_stride (`List[int]`, *optional*, defaults to `[2, 1, 1, 1]`):
            The strides for list of conv in stage 3.Should be of same length os `stage3_in_channels`
        stage4_in_channels (`List[int]`, *optional*, defaults to `[256, 512, 512, 512]`):
            The strides for list of conv in stage 4.Should be of same length os `stage4_in_channels`
        stage4_out_channels (`List[int]`, *optional*, defaults to `[512, 512, 512, 512]`):
            The num of channels in output for list of conv in stage 4.Should be of same length os `stage4_in_channels`
        stage4_kernel_size (`List[List[int]]`, *optional*, defaults to `[[3, 3], [3, 1], [1, 3], [3, 3]]`):
            The kernel sizes for list of conv in stage 4.Should be of same length os `stage4_in_channels`
        stage4_stride (`List[int]`, *optional*, defaults to `[2, 1, 1, 1]`):
            The strides for list of conv in stage 4.Should be of same length os `stage4_in_channels`
        hidden_sizes (`List[int]`, *optional*, defaults to `[64, 64, 128, 256, 512]`):
            Dimensionality (hidden size) at each stage.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        out_features (`List[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
            corresponding stages. If unset and `out_indices` is unset, will default to the last stage.
        out_indices (`List[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
            If unset and `out_features` is unset, will default to the last stage.

    Examples:

    ```python
    >>> from transformers import FastConfig, FastForSceneTextRecognition

    >>> # Initializing a Fast Config
    >>> configuration = FastConfig()

    >>> # Initializing a model (with random weights)
    >>> model = FastForSceneTextRecognition(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    r"""
    [Raghavan/fast_base_tt_800_finetune_ic17mlt](https://huggingface.co/Raghavan/fast_base_tt_800_finetune_ic17mlt)
    """
    model_type = "textnet"

    def __init__(
        self,
        kernel_size=3,
        stride=2,
        in_channels=3,
        out_channels=64,
        act_func="relu",
        stage1_in_channels=[64, 64, 64],
        stage1_out_channels=[64, 64, 64],
        stage1_kernel_size=[[3, 3], [3, 3], [3, 3]],
        stage1_stride=[1, 2, 1],
        stage2_in_channels=[64, 128, 128, 128],
        stage2_out_channels=[128, 128, 128, 128],
        stage2_kernel_size=[[3, 3], [1, 3], [3, 3], [3, 1]],
        stage2_stride=[2, 1, 1, 1],
        stage3_in_channels=[128, 256, 256, 256],
        stage3_out_channels=[256, 256, 256, 256],
        stage3_kernel_size=[[3, 3], [3, 3], [3, 1], [1, 3]],
        stage3_stride=[2, 1, 1, 1],
        stage4_in_channels=[256, 512, 512, 512],
        stage4_out_channels=[512, 512, 512, 512],
        stage4_kernel_size=[[3, 3], [3, 1], [1, 3], [3, 3]],
        stage4_stride=[2, 1, 1, 1],
        hidden_sizes=[64, 64, 128, 256, 512],
        initializer_range=0.02,
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_func = act_func

        self.stage1_in_channels = stage1_in_channels
        self.stage1_out_channels = stage1_out_channels
        self.stage1_kernel_size = stage1_kernel_size
        self.stage1_stride = stage1_stride

        self.stage2_in_channels = stage2_in_channels
        self.stage2_out_channels = stage2_out_channels
        self.stage2_kernel_size = stage2_kernel_size
        self.stage2_stride = stage2_stride

        self.stage3_in_channels = stage3_in_channels
        self.stage3_out_channels = stage3_out_channels
        self.stage3_kernel_size = stage3_kernel_size
        self.stage3_stride = stage3_stride

        self.stage4_in_channels = stage4_in_channels
        self.stage4_out_channels = stage4_out_channels
        self.stage4_kernel_size = stage4_kernel_size
        self.stage4_stride = stage4_stride

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
