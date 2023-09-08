# coding=utf-8
# Copyright 2023 IBM and HuggingFace Inc. team. All Rights Reserved.
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
""" PatchTSMixer model configuration"""

from typing import List, Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

PATCHTSMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ibm/patchtsmixer-base": "https://huggingface.co/ibm/patchtsmixer-base/resolve/main/config.json",
}



class PatchTSMixerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`PatchTSMixerModel`]. It is used to instantiate an
    PatchTSMixer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the PatchTSMixer
    [](https://huggingface.co/)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        seq_len (`int`, *optional*, defaults to 32):
            The context/history length for the input sequence.
        patch_len (`int`, *optional*, defaults to 8):
            The patch length for the `PatchTSMixer` model. Try to set it as a divisor of `seq_len`.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input variables.
        stride (`int`, *optional*, defaults to 8):
            Determines the overlap between two consecutive patches. 
        num_features (`int`, *optional*, defaults to 8):
            Hidden dimension of the model. Larger value indicates more complex model.
        expansion_factor (`int`, *optional*, defaults to 2):
            Expansion factor to use inside MLP. Larger value indicates more complex model.
        num_layers (`int`, *optional*, defaults to 2):
            Number of layers to use. Larger value indicates more complex model.
        dropout (`float`, *optional*, defaults to 0.2):
            The dropout probability the `PatchTSMixer` backbone.
        mode (`str`, *optional*, defaults to "common_channel"):
            Mixer Mode. Determines how to process the channels. Allowed values: "flatten", "common_channel", "mix_channel". 
            In "flatten" mode, patch embedding encodes the patch information across all channels.
            In "common_channel" mode, patch embedding is independent of channels (Channel Independece). 
            In "mix_channel" mode, we follow channel independence, but in addition to patch and feature mixing, 
            we also do channel mixing.
        gated_attn (`bool`, *optional*, defaults to `True`):
            Enable Gated Attention. 
        norm_mlp (`str`, *optional*, defaults to "LayerNorm"):
            Normalization layer (BatchNorm or LayerNorm).
        self_attn (`bool`, *optional*, defaults to `False`):
            Enable Tiny self attention in addition to MLP mixing.
        self_attn_heads (`int`, *optional*, defaults to 1):
            Number of self-attention heads. Works only when `self_attn` is set to `True`.
        use_pe (`bool`, *optional*, defaults to `False`):
            Enable the use of positional embedding for the tiny self-attention layers. 
            Works only when `self_attn` is set to `True`.
        pe (`str`, *optional*, defaults to "zeros"):
            Type of positional encoding. Allowed values are `None`, "zeros", "normal", "uniform", "sincos".
        learn_pe (`bool`, *optional*, defaults to `False`):
            Whether to learn the positional encoding.
        mask_type (`str`, *optional*, defaults to "random"):
            Type of masking for pretraining. Allowed values are "random", "forecast".  
        mask_ratio (`float`, *optional*, defaults to 0.5):
            Masking ratio. Higher value indicates more masking.
        mask_patches (`list`, *optional*, defaults to [2, 3]): 
            List of patch lengths to mask in the end of the data.
        mask_patch_ratios (`list`, *optional*, defaults to [1, 1]):  
            List of weights to use for each patch length. For Example, if `mask_patches` is [2,3] and 
            `mask_patch_ratios` is [1,1], then equal weights to both patch lengths.
        mask_value (`float`, *optional*, defaults to 0.0):
            Mask value to use.
        masked_loss (`bool`, *optional*, defaults to `False`):
            Whether to compute pretraining loss only at the masked portions, or on the entire output.
        channel_consistent_masking (`bool`, *optional*, defaults to `True`):
            When true, masking will be same across all channels of a timeseries. 
            Otherwise, masking positions will vary across channels. 
        revin (`bool`, *optional*, defaults to `True`):
            Whether to apply [Reversible Instance Normalization](https://openreview.net/pdf?id=cGDAkQo1C0p). 
        head_dropout (`float`, *optional*, defaults to 0.2):
            The dropout probability the `PatchTSMixer` head. 
        forecast_len (`int`, *optional*, defaults to 16):
            Number of time steps to forecast for a forecasting task. Also known as the Forecast Horizon.
        forecast_channel_indices (`list`, *optional*, defaults to `None`):
            List of channel indices to forecast. If None, forecast all channels.
        n_classes (`int`, *optional*, defaults to 3):
            Number of classes for a classification task.
        n_targets (`int`, *optional*, defaults to 3):
            Number of targets (dimensionality of the regressed variable) for a regression task. 
        output_range (`list`, *optional*, defaults to `None`):
            Output range to restrict. Defaults to None.
        head_agg (`str`, *optional*, defaults to "max_pool"):
            Aggregation mode. Allowed values are `None`, "use_last", "max_pool", "avg_pool".
        is_encoder_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as an encoder/decoder or not. Allowed value is `False`.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal weight initialization distribution.
        seed_number (`int`, *optional*, defaults to 42):
            Random seed.
        post_init (`bool`, *optional*, defaults to `False`):
            Whether to use custom weight initialization from `transformers` library, 
            or the default initialization in `PyTorch`. Setting it to `False` performs
            `PyTorch` weight initialization.
        
    Example:

    ```python
    >>> from transformers import PatchTSMixerConfig, PatchTSMixerModel

    >>> # Initializing a default PatchTSMixer configuration
    >>> configuration = PatchTSMixerConfig()

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = PatchTSMixerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "patchtsmixer"
    attribute_map = {
        "hidden_size": "num_features",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        seq_len: int = 32,
        patch_len: int = 8,
        in_channels: int = 3,
        stride: int = 8,
        num_features: int = 8,
        expansion_factor: int = 2,
        num_layers: int = 3,
        dropout: float = 0.2,
        mode: str = "common_channel",
        gated_attn: bool = True,
        norm_mlp: str="LayerNorm",
        self_attn: bool = False,
        self_attn_heads: int = 1,
        use_pe: bool = False,
        pe: str = "zeros",
        learn_pe: bool = False,
        mask_type: str = "random",
        mask_ratio=0.5,
        mask_patches: list = [2, 3],
        mask_patch_ratios: list = [1, 1],
        mask_value=0,
        masked_loss: bool = False,
        channel_consistent_masking: bool = True,
        revin: bool = True,
        head_dropout: float = 0.2,
        forecast_len: int = 16,
        forecast_channel_indices: list = None,
        n_classes: int = 3,
        n_targets: int = 3,
        output_range: list = None,
        head_agg: str = "max_pool",
        is_encoder_decoder: bool = False,
        init_std: float = 0.02,
        seed_number: int = 42,
        post_init: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_features = num_features
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers
        self.dropout = dropout
        self.mode = mode
        self.gated_attn = gated_attn
        
        self.norm_mlp = norm_mlp

        self.revin = revin
        
        self.head_dropout = head_dropout

        self.num_patches = (
            max(seq_len, patch_len) - patch_len
        ) // stride + 1

        # masking related
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.mask_patches = mask_patches
        self.mask_patch_ratios = mask_patch_ratios
        self.mask_value = mask_value
        self.channel_consistent_masking = channel_consistent_masking
        # self.mask_mode = mask_mode
        self.masked_loss = masked_loss
        # patching related
        self.patch_last = True

        self.use_pe = use_pe
        self.pe = pe
        self.learn_pe = learn_pe

        # forecast/prediction related
        self.forecast_len = forecast_len
        self.forecast_channel_indices = forecast_channel_indices

        # classification/regression related
        self.n_classes = n_classes
        self.n_targets = n_targets
        self.output_range = output_range
        self.head_agg = head_agg

        # other params
        self.self_attn = self_attn
        self.self_attn_heads = self_attn_heads
        self.d_size = "4D"
        self.init_std = init_std
        self.seed_number = seed_number
        self.post_init = post_init
        
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)
