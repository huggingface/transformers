# coding=utf-8
# Copyright 2024 Google AI and The HuggingFace Inc. team. All rights reserved.
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
"""ProPainter model configuration"""

from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class ProPainterConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ProPainterModel`]. It is used to instantiate an ProPainter
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ProPainter
    [ruffy369/propainter](https://huggingface.co/ruffy369/propainter) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        encoder_stride (`int`, *optional*, defaults to 16):
           Factor to increase the spatial resolution by in the decoder head for masked image modeling.

    Example:

    ```python
    >>> from transformers import ProPainterConfig, ProPainterModel

    >>> # Initializing a ProPainter propainter-base-patch16-224 style configuration
    >>> configuration = ProPainterConfig()

    >>> # Initializing a model (with random weights) from the propainter-base-patch16-224 style configuration
    >>> model = ProPainterModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "propainter"
    # attribute_map = {
    #     "hidden_size": "latent_dim",
    #     "num_attention_heads": "attention_heads",
    #     "num_hidden_layers": "num_enc_layers",
    #     "vocab_size": "vocab_size",
    #     "num_key_value_heads": "num_heads",
    # }


    def __init__(
        self,
        # hidden_size=768,
        # num_hidden_layers=12,
        # num_attention_heads=12,
        # intermediate_size=3072,
        # hidden_act="gelu",
        # hidden_dropout_prob=0.0,
        # attention_probs_dropout_prob=0.0,
        # initializer_range=0.02,
        # layer_norm_eps=1e-12,
        # image_size=224,
        # patch_size=16,
        # num_channels=3,
        # qkv_bias=True,
        # encoder_stride=16,
        width_flow_complete_net=432,
        height_flow_complete_net=240,
        width_propainter=432,
        height_propainter=240,
        num_local_frames_flow_complete_net=10,
        num_ref_frames_flow_complete_net=1,
        num_local_frames_propainter=10,
        num_ref_frames_propainter=6,
        load_flow_flow_complete_net=0,
        load_flow_propainter=0,
        flow_weight_flow_complete_net=0.25,
        beta1_flow_complete_net=0,
        beta2_flow_complete_net=0.99,
        lr_flow_complete_net=5e-5,
        batch_size_flow_complete_net=8,
        num_workers_flow_complete_net=4,
        num_prefetch_queue_flow_complete_net=4,
        log_freq_flow_complete_net=100,
        save_freq_flow_complete_net=5e3,
        iterations_flow_complete_net=700e3,
        hole_weight=1,
        valid_weight=1,
        flow_weight_propainter=1,
        adversarial_weight=0.01,
        GAN_LOSS="hinge",
        perceptual_weight=0,
        interp_mode="nearest",
        beta1_propainter=0,
        beta2_propainter=0.99,
        lr_propainter=1e-4,
        batch_size_propainter=8,
        num_workers_propainter=8,
        num_prefetch_queue_propainter=8,
        log_freq_propainter=100,
        save_freq_propainte=1e4,
        iterations_propainter=700e3,
        ref_stride=10,
        neighbor_length=10,
        subvideo_length=80,
        raft_optical_flow_iter=20,
        corr_levels=4,
        corr_radius=4,
        dropout=0,
        initializer_range=0.02,
        raft_iter=20,
        num_channels=128,
        hidden_size=512,
        kernel_size=(7, 7),
        padding=(3, 3),
        stride=(3, 3),
        num_hidden_layers=8,
        num_attention_heads=4,
        window_size=(5, 9),
        pool_size=(4, 4),
        no_dis=0,
        flow_weight=0.25,
        
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.width_flow_complete_net =  width_flow_complete_net
        self.height_flow_complete_net = height_flow_complete_net
        self.width_propainter =  width_propainter
        self.height_propainter = height_propainter
        self.num_local_frames_flow_complete_net = num_local_frames_flow_complete_net
        self.num_ref_frames_flow_complete_net = num_ref_frames_flow_complete_net
        self.num_local_frames_propainter = num_local_frames_propainter
        self.num_ref_frames_propainter = num_ref_frames_propainter
        self.load_flow_flow_complete_net = load_flow_flow_complete_net
        self.load_flow_propainter = load_flow_propainter
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
        self.GAN_LOSS = GAN_LOSS
        self.perceptual_weight = perceptual_weight
        self.interp_mode = interp_mode
        self.beta1_propainter = beta1_propainter
        self.beta2_propainter = beta2_propainter
        self.lr_propainter = lr_propainter
        self.batch_size_propainter = batch_size_propainter
        self.num_workers_propainter = num_workers_propainter
        self.num_prefetch_queue_propainter = num_prefetch_queue_propainter
        self.log_freq_propainter = log_freq_propainter
        self.save_freq_propainte = save_freq_propainte
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
        self.flow_weight = flow_weight