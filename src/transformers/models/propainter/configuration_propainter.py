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
        width_flow_complete_net (`int`, *optional*, defaults to 432):
            The width of the input for the flow completion network.
        height_flow_complete_net (`int`, *optional*, defaults to 240):
            The height of the input for the flow completion network.
        width_propainter (`int`, *optional*, defaults to 432):
            The width of the input for the ProPainter inpaint_generator network.
        height_propainter (`int`, *optional*, defaults to 240):
            The height of the input for the ProPainter inpaint_generator network.
        num_local_frames_flow_complete_net (`int`, *optional*, defaults to 10):
            The number of local frames used in the flow completion network.
        num_ref_frames_flow_complete_net (`int`, *optional*, defaults to 1):
            The number of reference frames used in the flow completion network.
        num_local_frames_propainter (`int`, *optional*, defaults to 10):
            The number of local frames used in the ProPainter inpaint_generator network.
        num_ref_frames_propainter (`int`, *optional*, defaults to 6):
            The number of reference frames used in the ProPainter inpaint_generator network.
        load_flow_in_flow_complete_net (`bool`, *optional*, defaults to `False`):
            Whether to load flow for the flow completion network.
        load_flow_in_propainter (`bool`, *optional*, defaults to `False`):
            Whether to load flow for the ProPainter inpaint_generator network.
        flow_weight_flow_complete_net (`float`, *optional*, defaults to 0.25):
            The weight of the flow loss in the flow completion network.
        beta1_flow_complete_net (`float`, *optional*, defaults to 0.0):
            The beta1 parameter for the optimizer in the flow completion network.
        beta2_flow_complete_net (`float`, *optional*, defaults to 0.99):
            The beta2 parameter for the optimizer in the flow completion network.
        lr_flow_complete_net (`float`, *optional*, defaults to 5e-5):
            The learning rate for the flow completion network.
        batch_size_flow_complete_net (`int`, *optional*, defaults to 8):
            The batch size for the flow completion network.
        num_workers_flow_complete_net (`int`, *optional*, defaults to 4):
            The number of workers for the flow completion network data loader.
        num_prefetch_queue_flow_complete_net (`int`, *optional*, defaults to 4):
            The number of prefetch queues for the flow completion network data loader.
        log_freq_flow_complete_net (`int`, *optional*, defaults to 100):
            The logging frequency for the flow completion network.
        save_freq_flow_complete_net (`int`, *optional*, defaults to 5000):
            The save frequency for the flow completion network, in terms of iterations.
        iterations_flow_complete_net (`int`, *optional*, defaults to 700000):
            The number of iterations for training the flow completion network.
        hole_weight (`float`, *optional*, defaults to 1.0):
            The weight for the hole loss.
        valid_weight (`float`, *optional*, defaults to 1.0):
            The weight for the valid region loss.
        flow_weight_propainter (`float`, *optional*, defaults to 1.0):
            The weight of the flow loss in the ProPainter inpaint_generator network.
        adversarial_weight (`float`, *optional*, defaults to 0.01):
            The weight of the adversarial loss in the ProPainter inpaint_generator network.
        gan_loss (`str`, *optional*, defaults to `"hinge"`):
            The type of GAN loss to use. Options are `"hinge"`, `"nsgan"`, or `"lsgan"`.
        perceptual_weight (`float`, *optional*, defaults to 0.0):
            The weight of the perceptual loss.
        interp_mode (`str`, *optional*, defaults to `"nearest"`):
            The interpolation mode used for resizing. Options are `"nearest"`, `"bilinear"`, etc.
        beta1_propainter (`float`, *optional*, defaults to 0.0):
            The beta1 parameter for the optimizer in the ProPainter inpaint_generator network.
        beta2_propainter (`float`, *optional*, defaults to 0.99):
            The beta2 parameter for the optimizer in the ProPainter inpaint_generator network.
        lr_propainter (`float`, *optional*, defaults to 1e-4):
            The learning rate for the ProPainter inpaint_generator network.
        batch_size_propainter (`int`, *optional*, defaults to 8):
            The batch size for the ProPainter inpaint_generator network.
        num_workers_propainter (`int`, *optional*, defaults to 8):
            The number of workers for the ProPainter inpaint_generator network data loader.
        num_prefetch_queue_propainter (`int`, *optional*, defaults to 8):
            The number of prefetch queues for the ProPainter inpaint_generator network data loader.
        log_freq_propainter (`int`, *optional*, defaults to 100):
            The logging frequency for the ProPainter inpaint_generator network.
        save_freq_propainter (`int`, *optional*, defaults to 10000):
            The save frequency for the ProPainter inpaint_generator network, in terms of iterations.
        iterations_propainter (`int`, *optional*, defaults to 700000):
            The number of iterations for training the ProPainter inpaint_generator network.
        ref_stride (`int`, *optional*, defaults to 10):
            The stride for reference frames in the ProPainter inpaint_generator network.
        neighbor_length (`int`, *optional*, defaults to 10):
            The length of neighboring frames considered in the ProPainter inpaint_generator network.
        subvideo_length (`int`, *optional*, defaults to 80):
            The length of sub-videos for training.
        raft_optical_flow_iter (`int`, *optional*, defaults to 20):
            The number of iterations for RAFT optical flow computation.
        corr_levels (`int`, *optional*, defaults to 4):
            The number of correlation levels used in the RAFT optical flow model.
        corr_radius (`int`, *optional*, defaults to 4):
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
        kernel_size (`List[int, int]`, *optional*, defaults to `[7, 7]`):
            The size of the convolution kernels.
        padding (`List[int, int]`, *optional*, defaults to `[3, 3]`):
            The padding size for the convolution kernels.
        stride (`List[int, int]`, *optional*, defaults to `[3, 3]`):
            The stride for the convolution kernels.
        num_hidden_layers (`int`, *optional*, defaults to 8):
            The number of hidden layers in the model.
        num_attention_heads (`int`, *optional*, defaults to 4):
            The number of attention heads for each attention layer in the model.
        window_size (`List[int, int]`, *optional*, defaults to `[5, 9]`):
            The size of the sliding window for attention operations.
        pool_size (`List[int, int]`, *optional*, defaults to `[4, 4]`):
            The size of the pooling layers in the model.
        no_dis (`bool`, *optional*, defaults to `False`):
            Whether to disable discriminator.
        in_channels (`List[int, int, int]`, *optional*, defaults to `[64, 64, 96]`):
            The number of input channels at different levels of the model.
        channels (`List[int, int, int]`, *optional*, defaults to `[64, 96, 128]`):
            The number of channels at different levels of the model.
        strides (`List[int, int, int]`, *optional*, defaults to `[1, 2, 2]`):
            The stride values for the convolution layers at different levels of the model.

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
        width_flow_complete_net=432,
        height_flow_complete_net=240,
        width_propainter=432,
        height_propainter=240,
        num_local_frames_flow_complete_net=10,
        num_ref_frames_flow_complete_net=1,
        num_local_frames_propainter=10,
        num_ref_frames_propainter=6,
        load_flow_in_flow_complete_net=False,
        load_flow_in_propainter=False,
        flow_weight_flow_complete_net=0.25,
        beta1_flow_complete_net=0.0,
        beta2_flow_complete_net=0.99,
        lr_flow_complete_net=5e-5,
        batch_size_flow_complete_net=8,
        num_workers_flow_complete_net=4,
        num_prefetch_queue_flow_complete_net=4,
        log_freq_flow_complete_net=100,
        save_freq_flow_complete_net=5000,
        iterations_flow_complete_net=700000,
        hole_weight=1.0,
        valid_weight=1.0,
        flow_weight_propainter=1.0,
        adversarial_weight=0.01,
        gan_loss="hinge",
        perceptual_weight=0.0,
        interp_mode="nearest",
        beta1_propainter=0.0,
        beta2_propainter=0.99,
        lr_propainter=1e-4,
        batch_size_propainter=8,
        num_workers_propainter=8,
        num_prefetch_queue_propainter=8,
        log_freq_propainter=100,
        save_freq_propainter=10000,
        iterations_propainter=700000,
        ref_stride=10,
        neighbor_length=10,
        subvideo_length=80,
        raft_optical_flow_iter=20,
        corr_levels=4,
        corr_radius=4,
        dropout=0.0,
        initializer_range=0.02,
        raft_iter=20,
        num_channels=128,
        hidden_size=512,
        kernel_size=[7, 7],
        padding=[3, 3],
        stride=[3, 3],
        num_hidden_layers=8,
        num_attention_heads=4,
        window_size=[5, 9],
        pool_size=[4, 4],
        no_dis=False,
        in_channels=[64, 64, 96],
        channels=[64, 96, 128],
        strides=[1, 2, 2],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.width_flow_complete_net = width_flow_complete_net
        self.height_flow_complete_net = height_flow_complete_net
        self.width_propainter = width_propainter
        self.height_propainter = height_propainter
        self.num_local_frames_flow_complete_net = num_local_frames_flow_complete_net
        self.num_ref_frames_flow_complete_net = num_ref_frames_flow_complete_net
        self.num_local_frames_propainter = num_local_frames_propainter
        self.num_ref_frames_propainter = num_ref_frames_propainter
        self.load_flow_in_flow_complete_net = load_flow_in_flow_complete_net
        self.load_flow_in_propainter = load_flow_in_propainter
        self.flow_weight_flow_complete_net = flow_weight_flow_complete_net
        self.beta1_flow_complete_net = beta1_flow_complete_net
        self.beta2_flow_complete_net = beta2_flow_complete_net
        self.lr_flow_complete_net = lr_flow_complete_net
        self.batch_size_flow_complete_net = batch_size_flow_complete_net
        self.num_workers_flow_complete_net = num_workers_flow_complete_net
        self.num_prefetch_queue_flow_complete_net = num_prefetch_queue_flow_complete_net
        self.log_freq_flow_complete_net = log_freq_flow_complete_net
        self.save_freq_flow_complete_net = save_freq_flow_complete_net
        self.iterations_flow_complete_net = iterations_flow_complete_net
        self.hole_weight = hole_weight
        self.valid_weight = valid_weight
        self.flow_weight_propainter = flow_weight_propainter
        self.adversarial_weight = adversarial_weight
        self.gan_loss = gan_loss
        self.perceptual_weight = perceptual_weight
        self.interp_mode = interp_mode
        self.beta1_propainter = beta1_propainter
        self.beta2_propainter = beta2_propainter
        self.lr_propainter = lr_propainter
        self.batch_size_propainter = batch_size_propainter
        self.num_workers_propainter = num_workers_propainter
        self.num_prefetch_queue_propainter = num_prefetch_queue_propainter
        self.log_freq_propainter = log_freq_propainter
        self.save_freq_propainter = save_freq_propainter
        self.iterations_propainter = iterations_propainter
        self.ref_stride = ref_stride
        self.neighbor_length = neighbor_length
        self.subvideo_length = subvideo_length
        self.raft_optical_flow_iter = raft_optical_flow_iter
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.raft_iter = raft_iter
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.window_size = window_size
        self.pool_size = pool_size
        self.no_dis = no_dis
        self.in_channels = in_channels
        self.channels = channels
        self.strides = strides
