# coding=utf-8
# Copyright 2024 S-Lab, Nanyang Technological University and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the S-Lab License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/sczhou/ProPainter/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ProPainter model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class ProPainterConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ProPainterModel`]. It is used to instantiate a ProPainter
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ProPainter [ruffy369/propainter](https://huggingface.co/ruffy369/propainter)
    architecture.

    The original configuration and code can be referred from [here](https://github.com/sczhou/ProPainter)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        num_local_frames_propainter (`int`, *optional*, defaults to 10):
            The number of local frames used in the ProPainter inpaint_generator network.
        flow_weight_flow_complete_net (`float`, *optional*, defaults to 0.25):
            The weight of the flow loss in the flow completion network.
        hole_weight (`float`, *optional*, defaults to 1.0):
            The weight for the hole loss.
        valid_weight (`float`, *optional*, defaults to 1.0):
            The weight for the valid region loss.
        adversarial_weight (`float`, *optional*, defaults to 0.01):
            The weight of the adversarial loss in the ProPainter inpaint_generator network.
        gan_loss (`str`, *optional*, defaults to `"hinge"`):
            The type of GAN loss to use. Options are `"hinge"`, `"nsgan"`, or `"lsgan"`.
        perceptual_weight (`float`, *optional*, defaults to 0.0):
            The weight of the perceptual loss.
        interp_mode (`str`, *optional*, defaults to `"nearest"`):
            The interpolation mode used for resizing. Options are `"nearest"`, `"bilinear"`, `"bicubic"`.
        ref_stride (`int`, *optional*, defaults to 10):
            The stride for reference frames in the ProPainter inpaint_generator network.
        neighbor_length (`int`, *optional*, defaults to 10):
            The length of neighboring frames considered in the ProPainter inpaint_generator network.
        subvideo_length (`int`, *optional*, defaults to 80):
            The length of sub-videos for training.
        correlation_levels (`int`, *optional*, defaults to 4):
            The number of correlation levels used in the RAFT optical flow model.
        correlation_radius (`int`, *optional*, defaults to 4):
            The radius of the correlation window used in the RAFT optical flow model.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability applied to layers in the model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing weight matrices.
        raft_iter (`int`, *optional*, defaults to 20):
            The number of iterations for RAFT model updates.
        num_channels (`int`, *optional*, defaults to 128):
            The number of channels in the feature maps.
        hidden_size (`int`, *optional*, defaults to 512):
            The dimensionality of hidden layers.
        kernel_size (`List[int]`, *optional*, defaults to `[7, 7]`):
            The size of the convolution kernels.
        kernel_size_3d (`List[int]`, *optional*, defaults to `[1, 3, 3]`):
            The size of the 3d convolution kernels.
        kernel_size_3d_discriminator (`List[int]`, *optional*, defaults to `[3, 5, 5]`):
            The size of the 3d convolution kernels for discriminator modules used to calculate losses.
        padding_inpaint_generator (`List[int]`, *optional*, defaults to `[3, 3]`):
            The padding size for the convolution kernels in inpaint_generator module.
        padding (`int`, *optional*, defaults to 1):
            The padding size for the convolution kernels.
        stride (`List[int]`, *optional*, defaults to `[3, 3]`):
            The stride for the convolution kernels.
        stride_3d (`List[int]`, *optional*, defaults to `[1, 1, 1]`):
            The stride for the 3d convolution kernels.
        num_hidden_layers (`int`, *optional*, defaults to 8):
            The number of hidden layers in the model.
        num_attention_heads (`int`, *optional*, defaults to 4):
            The number of attention heads for each attention layer in the model.
        window_size (`List[int]`, *optional*, defaults to `[5, 9]`):
            The size of the sliding window for attention operations.
        pool_size (`List[int]`, *optional*, defaults to `[4, 4]`):
            The size of the pooling layers in the model.
        no_dis (`bool`, *optional*, defaults to `False`):
            Whether to disable discriminator.
        in_channels (`List[int]`, *optional*, defaults to `[64, 64, 96]`):
            The number of input channels at different levels of the model.
        channels (`List[int]`, *optional*, defaults to `[64, 96, 128]`):
            The number of channels at different levels of the model.
        strides (`List[int]`, *optional*, defaults to `[1, 2, 2]`):
            The stride values for the convolution layers at different levels of the model.
        norm_fn (`List[str]`, *optional*, defaults to `['batch', 'group', 'instance', 'none']`):
            The type of normalization to use in the model. Available options are:
            - `"batch"`: Use Batch Normalization.
            - `"group"`: Use Group Normalization.
            - `"instance"`: Use Instance Normalization.
            - `"none"`: No normalization will be applied.
        patch_size (`int`, *optional*, defaults to 3):
            The kernel size of the 2D convolution layer.

    Example:

    ```python
    >>> from transformers import ProPainterConfig, ProPainterModel

    >>> # Initializing a ProPainter style configuration
    >>> configuration = ProPainterConfig()

    >>> # Initializing a model (with random weights) from the propainter style configuration
    >>> model = ProPainterModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "propainter"

    def __init__(
        self,
        num_local_frames_propainter=10,
        flow_weight_flow_complete_net=0.25,
        hole_weight=1.0,
        valid_weight=1.0,
        adversarial_weight=0.01,
        gan_loss="hinge",
        perceptual_weight=0.0,
        interp_mode="nearest",
        ref_stride=10,
        neighbor_length=10,
        subvideo_length=80,
        correlation_levels=4,
        correlation_radius=4,
        dropout=0.0,
        initializer_range=0.02,
        raft_iter=20,
        num_channels=128,
        hidden_size=512,
        kernel_size=[7, 7],
        kernel_size_3d=[1, 3, 3],
        kernel_size_3d_discriminator=[3, 5, 5],
        padding_inpaint_generator=[3, 3],
        padding=1,
        stride=[3, 3],
        stride_3d=[1, 1, 1],
        num_hidden_layers=8,
        num_attention_heads=4,
        window_size=[5, 9],
        pool_size=[4, 4],
        no_dis=False,
        in_channels=[64, 64, 96],
        channels=[64, 96, 128],
        strides=[1, 2, 2],
        norm_fn=["batch", "group", "instance", "none"],
        patch_size=3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_local_frames_propainter = num_local_frames_propainter
        self.flow_weight_flow_complete_net = flow_weight_flow_complete_net
        self.hole_weight = hole_weight
        self.valid_weight = valid_weight
        self.adversarial_weight = adversarial_weight
        self.gan_loss = gan_loss
        self.perceptual_weight = perceptual_weight
        self.interp_mode = interp_mode
        self.ref_stride = ref_stride
        self.neighbor_length = neighbor_length
        self.subvideo_length = subvideo_length
        self.correlation_levels = correlation_levels
        self.correlation_radius = correlation_radius
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.raft_iter = raft_iter
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.kernel_size_3d = kernel_size_3d
        self.kernel_size_3d_discriminator = kernel_size_3d_discriminator
        self.padding_inpaint_generator = padding_inpaint_generator
        self.padding = padding
        self.stride = stride
        self.stride_3d = stride_3d
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.window_size = window_size
        self.pool_size = pool_size
        self.no_dis = no_dis
        self.in_channels = in_channels
        self.channels = channels
        self.strides = strides
        self.norm_fn = norm_fn
        self.patch_size = patch_size
