# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""CHMv2 model configuration"""

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


class CHMv2Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CHMv2Model`]. It is used to instantiate a CHMv2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the CHMv2 model.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        backbone_config (`Union[dict, "PreTrainedConfig"]`, *optional*):
            The configuration of the backbone model. Can be DINOv3ViTConfig.
        backbone_type (`str`, *optional*, defaults to `"dinov3_vitl16"`):
            The type of backbone to use.
        patch_size (`int`, *optional*, defaults to 16):
            The size of the patches to extract from the backbone features.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        reassemble_hidden_size (`int`, *optional*, defaults to 1024):
            The number of input channels of the reassemble layers.
        reassemble_factors (`list[int]`, *optional*, defaults to `[4, 2, 1, 0.5]`):
            The up/downsampling factors of the reassemble layers.
        neck_hidden_sizes (`list[str]`, *optional*, defaults to `[128, 256, 512, 1024]`):
            The output channel sizes of the reassemble stage for each backbone feature level.
            These correspond to post_process_channels in the CHMv2Head.
        fusion_hidden_size (`int`, *optional*, defaults to 256):
            The number of channels before fusion.
        head_in_index (`int`, *optional*, defaults to -1):
            The index of the features to use in the depth estimation head.
        head_hidden_size (`int`, *optional*, defaults to 128):
            The number of channels in the hidden layer of the depth estimation head.
        n_output_channels (`int`, *optional*, defaults to 256):
            Number of output channels for the CHMv2 head (number of depth bins).
        use_batchnorm (`bool`, *optional*, defaults to `False`):
            Whether to use batchnorm in the ReassembleBlocks.
        use_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the FeatureFusionBlocks.
        readout_type (`str`, *optional*, defaults to `"project"`):
            Type of readout operation. Can be one of `["ignore", "add", "project"]`.
        expand_channels (`bool`, *optional*, defaults to `False`):
            Whether to expand the channels in post process block.
        min_depth (`float`, *optional*, defaults to 0.001):
            The minimum depth value for depth bin calculation.
        max_depth (`float`, *optional*, defaults to 80.0):
            The maximum depth value for depth bin calculation.
        bins_strategy (`str`, *optional*, defaults to `"chmv2_mixlog"`):
            The strategy for depth bins distribution. Can be one of `["linear", "log", "chmv2_mixlog"]`.
        norm_strategy (`str`, *optional*, defaults to `"chmv2_mixlog"`):
            The normalization strategy for depth prediction. Can be one of `["linear", "softmax", "sigmoid", "chmv2_mixlog"]`.

    Example:

    ```python
    >>> from transformers import CHMv2Config, CHMv2ForCanopyHeightEstimation

    >>> # Initializing a CHMv2 style configuration with DINOv3 backbone
    >>> configuration = CHMv2Config(backbone_type="dinov3_vitl16")

    >>> # Initializing a model from the CHMv2 style configuration
    >>> model = CHMv2ForCanopyHeightEstimation(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "chmv2"
    sub_configs = {"backbone_config": AutoConfig}

    def __init__(
        self,
        backbone_config=None,
        backbone_type="dinov3_vitl16",
        patch_size=16,
        initializer_range=0.02,
        reassemble_hidden_size=1024,
        reassemble_factors=[4, 2, 1, 0.5],
        neck_hidden_sizes=[128, 256, 512, 1024],
        fusion_hidden_size=256,
        head_in_index=-1,
        head_hidden_size=128,
        n_output_channels=256,
        use_batchnorm=False,
        use_bias=True,
        readout_type="project",
        expand_channels=False,
        min_depth=0.001,
        max_depth=96.0,
        bins_strategy="chmv2_mixlog",
        norm_strategy="chmv2_mixlog",
        **kwargs,
    ):
        self.backbone_type = backbone_type
        supported_types = ["dinov3_vit", "dinov3_vitl16"]
        assert backbone_type in supported_types, f"{backbone_type} not supported. Choose from {supported_types}"

        default_config_kwargs = {
                "image_size": 416,
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_attention_heads": 16,
                "num_hidden_layers": 24,
                "num_register_tokens": 4,
                "key_bias": True,
                "out_indices": [5, 11, 17, 23],
                "reshape_hidden_states": True,
                "apply_layernorm": True,
                "layer_norm_eps": 1e-6,
            }

        backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=backbone_config,
            default_config_type="dinov3_vit",
            default_config_kwargs=default_config_kwargs,
            **kwargs,
        )

        self.backbone_config = backbone_config
        self.backbone_type = backbone_type
        self.reassemble_hidden_size = reassemble_hidden_size
        self.patch_size = patch_size
        self.initializer_range = initializer_range
        self.reassemble_factors = reassemble_factors
        self.neck_hidden_sizes = neck_hidden_sizes
        self.fusion_hidden_size = fusion_hidden_size
        self.head_in_index = head_in_index
        self.head_hidden_size = head_hidden_size
        self.n_output_channels = n_output_channels
        self.use_batchnorm = use_batchnorm
        self.use_bias = use_bias
        self.readout_type = readout_type
        self.expand_channels = expand_channels

        if bins_strategy not in ["linear", "log", "chmv2_mixlog"]:
            raise ValueError("bins_strategy must be one of ['linear', 'log', 'chmv2_mixlog']")
        if norm_strategy not in ["linear", "softmax", "sigmoid", "chmv2_mixlog"]:
            raise ValueError("norm_strategy must be one of ['linear', 'softmax', 'sigmoid', 'chmv2_mixlog']")

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.bins_strategy = bins_strategy
        self.norm_strategy = norm_strategy

        super().__init__(**kwargs)


__all__ = ["CHMv2Config"]
