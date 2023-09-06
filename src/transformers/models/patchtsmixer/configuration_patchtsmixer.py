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
        prediction_length (`int`):
            The prediction length for the decoder. In other words, the prediction horizon of the model.
        context_length (`int`, *optional*, defaults to `prediction_length`):
            The context length for the encoder. If unset, the context length will be the same as the
            `prediction_length`.
        distribution_output (`string`, *optional*, defaults to `"student_t"`):
            The distribution emission head for the model. Could be either "student_t", "normal" or "negative_binomial".
        loss (`string`, *optional*, defaults to `"nll"`):
            The loss function for the model corresponding to the `distribution_output` head. For parametric
            distributions it is the negative log likelihood (nll) - which currently is the only supported one.
        input_size (`int`, *optional*, defaults to 1):
            The size of the target variable which by default is 1 for univariate targets. Would be > 1 in case of
            multivariate targets.
        lags_sequence (`list[int]`, *optional*, defaults to `[1, 2, 3, 4, 5, 6, 7]`):
            The lags of the input time series as covariates often dictated by the frequency. Default is `[1, 2, 3, 4,
            5, 6, 7]`.
        scaling (`bool`, *optional* defaults to `True`):
            Whether to scale the input targets.
        num_time_features (`int`, *optional*, defaults to 0):
            The number of time features in the input time series.
        num_dynamic_real_features (`int`, *optional*, defaults to 0):
            The number of dynamic real valued features.
        num_static_categorical_features (`int`, *optional*, defaults to 0):
            The number of static categorical features.
        num_static_real_features (`int`, *optional*, defaults to 0):
            The number of static real valued features.
        cardinality (`list[int]`, *optional*):
            The cardinality (number of different values) for each of the static categorical features. Should be a list
            of integers, having the same length as `num_static_categorical_features`. Cannot be `None` if
            `num_static_categorical_features` is > 0.
        embedding_dimension (`list[int]`, *optional*):
            The dimension of the embedding for each of the static categorical features. Should be a list of integers,
            having the same length as `num_static_categorical_features`. Cannot be `None` if
            `num_static_categorical_features` is > 0.
        d_model (`int`, *optional*, defaults to 64):
            Dimensionality of the transformer layers.
        encoder_layers (`int`, *optional*, defaults to 2):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 2):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the Transformer decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 32):
            Dimension of the "intermediate" (often named feed-forward) layer in encoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 32):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and decoder. If string, `"gelu"` and
            `"relu"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the encoder, and decoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention and fully connected layers for each encoder layer.
        decoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention and fully connected layers for each decoder layer.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability used between the two layers of the feed-forward networks.
        num_parallel_samples (`int`, *optional*, defaults to 100):
            The number of samples to generate in parallel for each time step of inference.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal weight initialization distribution.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use the past key/values attentions (if applicable to the model) to speed up decoding.
        label_length (`int`, *optional*, defaults to 10):
            Start token length of the PatchTSMixer decoder, which is used for direct multi-step prediction (i.e.
            non-autoregressive generation).
        moving_average (`int`, defaults to 25):
            The window size of the moving average. In practice, it's the kernel size in AvgPool1d of the Decomposition
            Layer.
        autocorrelation_factor (`int`, defaults to 3):
            "Attention" (i.e. AutoCorrelation mechanism) factor which is used to find top k autocorrelations delays.
            It's recommended in the paper to set it to a number between 1 and 5.


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
