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

from typing import Optional, Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

PATCHTSMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ibm/patchtsmixer-etth1-pretrain": "https://huggingface.co/ibm/patchtsmixer-etth1-pretrain/resolve/main/config.json",
}


class PatchTSMixerConfig(PretrainedConfig):
    r"""
    PatchTSMixer is a lightweight time-series modelling backbone based on the MLP-Mixer architecture. The [KDD 2023
    paper](https://arxiv.org/abs/2306.09364) demonstrates its state-of-the-art performance for time series tasks, with
    reduced computational and memory demands. PatchTSMixer's distinguishing feature lies in its capacity to
    effortlessly facilitate lightweight mixing across patches, channels, and hidden features. This unique capability
    allows for the efficient modeling of dependencies within and across various dimensions. It also supports various
    attention mechanisms starting from simple gated attention to more complex self-attention blocks that can be enabled
    based on the input data requirements. PatchTSMixer backbone can currently be extended for various downstream tasks
    such as forecasting, classification and regression.

    This is the configuration class to store the configuration of an [`PatchTSMixerModel`]. It is used to instantiate a
    PatchTSMixer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the PatchTSMixer
    [ibm/patchtsmixer-etth1-pretrain](https://huggingface.co/ibm/patchtsmixer-etth1-pretrain) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        seq_len (`int`, *optional*, defaults to 32):
            The context/history length for the input sequence.
        patch_len (`int`, *optional*, defaults to 8):
            The patch length for the `PatchTSMixer` model. Try to set it as a divisor of `seq_len`.
        input_size (`int`, *optional*, defaults to 1):
            Number of input variates. For Univariate, set it to 1.
        stride (`int`, *optional*, defaults to 8):
            Determines the overlap between two consecutive patches. Set it to patch_len (or greater), if we want
            non-overlapping patches.
        num_features (`int`, *optional*, defaults to 8):
            Hidden dimension of the model. Recommended to set it as a multiple of patch_len (i.e. 2-5X of patch_len).
            Larger value indicates more complex model.
        expansion_factor (`int`, *optional*, defaults to 2):
            Expansion factor to use inside MLP. Recommended range is 2-5. Larger value indicates more complex model.
        num_layers (`int`, *optional*, defaults to 3):
            Number of layers to use. Recommended range is 3-15. Larger value indicates more complex model.
        dropout (`float`, *optional*, defaults to 0.2):
            The dropout probability the `PatchTSMixer` backbone. Recommended range is 0.2-0.7
        mode (`str`, *optional*, defaults to `"common_channel"`):
            Mixer Mode. Determines how to process the channels. Allowed values: "flatten", "common_channel",
            "mix_channel". In "flatten" mode, patch embedding encodes the patch information across all channels. (not a
            preferred approach) In "common_channel" mode, we follow Channel-independent modelling with no explicit
            channel-mixing. Channel mixing happens in an implict manner via shared weights across channels. (preferred
            first approach) In "mix_channel" mode, we follow explicit channel-mixing in addition to patch and feature
            mixer. (preferred approach when channel correlations are very important to model)
        gated_attn (`bool`, *optional*, defaults to `True`):
            Enable Gated Attention.
        norm_mlp (`str`, *optional*, defaults to `"LayerNorm"`):
            Normalization layer (BatchNorm or LayerNorm).
        self_attn (`bool`, *optional*, defaults to `False`):
            Enable Tiny self attention across patches. This can be enabled when the output of Vanilla PatchTSMixer with
            gated attention is not satisfactory. Enabling this leads to explicit pair-wise attention and modelling
            across patches.
        self_attn_heads (`int`, *optional*, defaults to 1):
            Number of self-attention heads. Works only when `self_attn` is set to `True`.
        use_positional_encoding (`bool`, *optional*, defaults to `False`):
            Enable the use of positional embedding for the tiny self-attention layers. Works only when `self_attn` is
            set to `True`.
        positional_encoding (`str`, *optional*, defaults to `"zeros"`):
            Type of positional encoding. Allowed values are `None`, "zeros", "normal", "uniform", "sincos". Works only
            when `use_positional_encoding` is set to `True`
        learn_positional_encoding (`bool`, *optional*, defaults to `False`):
            Whether to learn the positional encoding. Works only when `use_positional_encoding` is set to `True`
        mask_type (`str`, *optional*, defaults to `"random"`):
            Type of masking to use for Masked Pretraining mode. Allowed values are "random", "forecast". In Random
            masking, points are maskes random. In Forecast masking, Points are masked towards the end.
        mask_ratio (`float`, *optional*, defaults to 0.5):
            Masking ratio to use when `mask_type` is `random`. Higher value indicates more masking.
        mask_patches (`list`, *optional*, defaults to `[2, 3]`):
            List of patch lengths to mask in the end of the data, when `mask_type` is `forecast`.
        mask_patch_ratios (`list`, *optional*, defaults to `[1, 1]`):
            List of weights to use for each patch length for forecast masking. For Example, if `mask_patches` is [2,3]
            and `mask_patch_ratios` is [1,1], then equal weights to both patch lengths.
        mask_value (`float`, *optional*, defaults to `0.0`):
            Mask value to use.
        masked_loss (`bool`, *optional*, defaults to `True`):
            Whether to compute pretraining loss only at the masked portions, or on the entire output.
        channel_consistent_masking (`bool`, *optional*, defaults to `True`):
            When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary
            across channels.
        scaling (`string` or `bool`, *optional*, defaults to `"std"`):
            Whether to scale the input targets via "mean" scaler, "std" scaler or no scaler if `None`. If `True`, the
            scaler is set to "mean".
        head_dropout (`float`, *optional*, defaults to 0.2):
            The dropout probability the `PatchTSMixer` head.
        forecast_len (`int`, *optional*, defaults to 16):
            Number of time steps to forecast for a forecasting task. Also known as the Forecast Horizon.
        forecast_channel_indices (`list`, *optional*):
            List of channel indices to forecast. If None, forecast all channels. Target data is expected to have all
            channels and we explitly filter the channels in prediction and target before loss computation.
        num_targets (`int`, *optional*, defaults to 3):
            Number of targets (dimensionality of the regressed variable) for a regression task.
        output_range (`list`, *optional*):
            Output range to restrict for the regression task. Defaults to None.
        head_agg (`str`, *optional*, defaults to `"max_pool"`):
            Aggregation mode to enable for classification or regression task. Allowed values are `None`, "use_last",
            "max_pool", "avg_pool".
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal weight initialization distribution.
        seed_number (`int`, *optional*):
            Random seed for masking.
        post_init (`bool`, *optional*, defaults to `False`):
            Whether to use custom weight initialization from `transformers` library, or the default initialization in
            `PyTorch`. Setting it to `False` performs `PyTorch` weight initialization.
        distribution_output (`string`, *optional*, defaults to `"student_t"`):
            The distribution emission head for the model when loss is "nll". Could be either "student_t", "normal" or
            "negative_binomial".
        loss (`string`, *optional*, defaults to `"mse"`):
            The loss function for the model corresponding to the `distribution_output` head. For parametric
            distributions it is the negative log likelihood ("nll") and for point estimates it is the mean squared
            error "mse".
        num_parallel_samples (`int`, *optional*, defaults to 100):
            The number of samples to generate in parallel for probablistic forecast.

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
        input_size: int = 1,
        stride: int = 8,
        num_features: int = 8,
        expansion_factor: int = 2,
        num_layers: int = 3,
        dropout: float = 0.2,
        mode: str = "common_channel",
        gated_attn: bool = True,
        norm_mlp: str = "LayerNorm",
        self_attn: bool = False,
        self_attn_heads: int = 1,
        use_positional_encoding: bool = False,
        positional_encoding: str = "zeros",
        learn_positional_encoding: bool = False,
        mask_type: str = "random",
        mask_ratio=0.5,
        mask_patches: list = [2, 3],
        mask_patch_ratios: list = [1, 1],
        mask_value=0,
        masked_loss: bool = True,
        channel_consistent_masking: bool = True,
        scaling: Optional[Union[str, bool]] = "std",
        head_dropout: float = 0.2,
        forecast_len: int = 16,
        forecast_channel_indices: list = None,
        num_targets: int = 3,
        output_range: list = None,
        head_agg: str = "max_pool",
        init_std: float = 0.02,
        seed_number: Optional[int] = None,
        post_init: bool = False,
        distribution_output: str = "student_t",
        loss: str = "mse",
        num_parallel_samples: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
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

        self.scaling = scaling

        self.head_dropout = head_dropout

        self.num_patches = (max(seq_len, patch_len) - patch_len) // stride + 1

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

        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding = positional_encoding
        self.learn_positional_encoding = learn_positional_encoding

        # forecast/prediction related
        self.forecast_len = forecast_len
        self.forecast_channel_indices = forecast_channel_indices

        # classification/regression related
        self.num_targets = num_targets
        self.output_range = output_range
        self.head_agg = head_agg

        # other params
        self.self_attn = self_attn
        self.self_attn_heads = self_attn_heads
        self.init_std = init_std
        self.seed_number = seed_number
        self.post_init = post_init
        self.distribution_output = distribution_output
        self.loss = loss
        self.num_parallel_samples = num_parallel_samples

        super().__init__(**kwargs)
