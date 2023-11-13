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
"""PatchTST model configuration"""

from typing import List, Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

PATCHTST_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ibm/patchtst-base": "https://huggingface.co/ibm/patchtst-base/resolve/main/config.json",
    # See all PatchTST models at https://huggingface.co/ibm/models?filter=patchtst
}


class PatchTSTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`PatchTSTModel`]. It is used to instantiate an
    PatchTST model according to the specified arguments, defining the model architecture.
    [ibm/patchtst](https://huggingface.co/ibm/patchtst) architecture.

    Configuration objects inherit from [`PretrainedConfig`] can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_input_channels (`int`, *optional*, defaults to 1):
            The size of the target variable which by default is 1 for univariate targets. Would be > 1 in case of
            multivariate targets.
        context_length (`int`, *optional*, defaults to 32):
            The context length for the encoder.
        distribution_output (`str`, *optional*, defaults to `"student_t"`):
            The distribution emission head for the model when loss is "nll". Could be either "student_t", "normal" or
            "negative_binomial".
        loss (`str`, *optional*, defaults to `"mse"`):
            The loss function for the model corresponding to the `distribution_output` head. For parametric
            distributions it is the negative log likelihood ("nll") and for point estimates it is the mean squared
            error "mse".
        patch_length (`int`, *optional*, defaults to 1):
            Define the patch length of the patchification process.
        patch_stride (`int`, *optional*, defaults to 1):
            define the stride of the patchification process.
        encoder_layers (`int`, *optional*, defaults to 3):
            Number of encoder layers.
        d_model (`int`, *optional*, defaults to 64):
            Dimensionality of the transformer layers.
        encoder_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads for each attention layer in the Transformer encoder.
        shared_embedding (`bool`, *optional*, defaults to `True`):
            Sharing the input embedding across all channels.
        channel_attention (`bool`, *optional*, defaults to `False`):
            Activate channel attention block in the Transformer to allow channels to attend each other.
        encoder_ffn_dim (`int`, *optional*, defaults to 256):
            Dimension of the "intermediate" (often named feed-forward) layer in encoder.
        norm (`str` , *optional*, defaults to `"BatchNorm"`):
            Normalization at each Transformer layer. Can be `"BatchNorm"` or `"LayerNorm"`.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            A value added to the denominator for numerical stability of normalization.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention probabilities.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the encoder, and decoder.
        positional_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability in the positional embedding layer.
        dropout_path (`float`, *optional*, defaults to 0.0):
            The dropout path in the residual block.
        ff_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability used between the two layers of the feed-forward networks.
        bias (`bool`, *optional*, defaults to `True`):
            Consider bias in the feed-forward networks.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (string) in the encoder.`"gelu"` and `"relu"` are supported.
        pre_norm (`bool`, *optional*, defaults to `True`):
            Normalization is applied before self-attention if pre_norm is set to `True`. Otherwise, normalization is
            applied after residual block.
        positional_encoding_type (`str`, *optional*, defaults to `"sincos"`):
            Positional encodings. `"zeros"`, `"normal"`, `"uniform"' and `"sincos"` are supported.
        learn_pe (`bool`, *optional*, defaults to `False`):
            Whether the positional encoding is updated during training.
        use_cls_token (`bool`, *optional*, defaults to `False`):
            Whether cls token is used.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal weight initialization distribution.
        shared_projection (`bool`, *optional*, defaults to `True`):
            Sharing the projection layer across different channels in the forecast head.
        seed_number (`Optional`, *optional*):
            Seed number used for random masking. If unset, no seed is set.
        scaling (`Union`, *optional*, defaults to `"mean"`):
            Whether to scale the input targets via "mean" scaler, "std" scaler or no scaler if `None`. If `True`, the
            scaler is set to "mean".
        mask_input (`bool`, *optional*, defaults to `False`):
            Apply masking during the pretraining.
        mask_type (`str`, *optional*, defaults to `"random"`):
            Masking type. Only `"random"` and `"forecast"` are currently supported.
        random_mask_ratio (`float`, *optional*, defaults to 0.5):
            Masking ratio is applied to mask the input data during random pretraining.
        forecast_mask_patches (`List`, *optional*, defaults to `[2, 3]`):
            List of patch lengths to mask in the end of the data.
        forecast_mask_ratios (`List`, *optional*, defaults to `[1, 1]`):
            List of weights to use for each patch length. For Ex. if patch_lengths is [5,4] and mix_ratio is [1,1],
            then equal weights to both patch lengths. Defaults to None.
        channel_consistent_masking (`bool`, *optional*, defaults to `False`):
            If channel consistent masking is True, all the channels will have the same masking.
        unmasked_channel_indices (`list`, *optional*):
            Channels that are not masked during pretraining.
        mask_value (`int`, *optional*, defaults to 0):
            Define the value of entries to be masked when pretraining.
        pooling_type (`str`, *optional*, defaults to `"mean"`):
            Pooling of the embedding. `"mean"`, `"max"` and `None` are supported.
        head_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for head.
        prediction_length (`int`, *optional*, defaults to 24):
            The prediction length for the encoder. In other words, the prediction horizon of the model.
        num_targets (`int`, *optional*, defaults to 1):
            Number of targets for regression and classificastion tasks. For classification, it is the number of
            classes.
        output_range (`list`, *optional*):
            Output range for regression task. The range of output values can be set to enforce the model to produce
            values within a range.
        num_parallel_samples (`int`, *optional*, defaults to 100):
            The number of samples is generated in parallel for probablistic prediction.


    ```python
    >>> from transformers import PatchTSTConfig, PatchTSTModel

    >>> # Initializing an PatchTST configuration with 12 time steps for prediction
    >>> configuration = PatchTSTConfig(prediction_length=12)

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = PatchTSTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "patchtst"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
    }

    def __init__(
        self,
        # time series specific configuration
        num_input_channels: int = 1,
        context_length: int = 32,
        distribution_output: str = "student_t",
        loss: str = "mse",
        # PatchTST arguments
        patch_length: int = 1,
        patch_stride: int = 1,
        # Transformer architecture configuration
        encoder_layers: int = 3,
        d_model: int = 64,
        encoder_attention_heads: int = 4,
        shared_embedding: bool = True,
        channel_attention: bool = False,
        encoder_ffn_dim: int = 256,
        norm: str = "BatchNorm",
        norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        positional_dropout: float = 0.0,
        dropout_path: float = 0.0,
        ff_dropout: float = 0.0,
        bias: bool = True,
        activation_function: str = "gelu",
        pre_norm: bool = True,
        positional_encoding_type: str = "sincos",
        learn_pe: bool = False,
        use_cls_token: bool = False,
        init_std: float = 0.02,
        shared_projection: bool = True,
        seed_number: Optional[int] = None,
        scaling: Optional[Union[str, bool]] = "mean",
        # mask pretraining
        mask_input: Optional[bool] = None,
        mask_type: str = "random",
        random_mask_ratio: float = 0.5,
        forecast_mask_patches: List[int] = [2, 3],
        forecast_mask_ratios: List[int] = [1, 1],
        channel_consistent_masking: bool = False,
        unmasked_channel_indices: Optional[List[int]] = None,
        mask_value=0,
        # head
        pooling_type: str = "mean",
        head_dropout: float = 0.0,
        prediction_length: int = 24,
        num_targets: int = 1,
        output_range: List = None,
        # distribution head
        num_parallel_samples: int = 100,
        **kwargs,
    ):
        # time series specific configuration
        self.context_length = context_length
        self.num_input_channels = num_input_channels  # n_vars
        self.loss = loss
        self.distribution_output = distribution_output
        self.num_parallel_samples = num_parallel_samples

        # Transformer architecture configuration
        self.d_model = d_model
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.shared_embedding = shared_embedding
        self.channel_attention = channel_attention
        self.norm = norm
        self.norm_eps = norm_eps
        self.positional_dropout = positional_dropout
        self.dropout_path = dropout_path
        self.ff_dropout = ff_dropout
        self.bias = bias
        self.activation_function = activation_function
        self.pre_norm = pre_norm
        self.positional_encoding_type = positional_encoding_type
        self.learn_pe = learn_pe
        self.use_cls_token = use_cls_token
        self.init_std = init_std
        self.scaling = scaling

        # PatchTST parameters
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.num_patches = self._num_patches()

        # Mask pretraining
        self.seed_number = seed_number
        self.mask_input = mask_input
        self.mask_type = mask_type
        self.random_mask_ratio = random_mask_ratio  # for random masking
        self.forecast_mask_patches = forecast_mask_patches  # for forecast masking
        self.forecast_mask_ratios = forecast_mask_ratios
        self.channel_consistent_masking = channel_consistent_masking
        self.unmasked_channel_indices = unmasked_channel_indices
        self.mask_value = mask_value

        # general head params
        self.pooling_type = pooling_type
        self.head_dropout = head_dropout

        # For prediction head
        self.shared_projection = shared_projection
        self.prediction_length = prediction_length

        # For prediction and regression head
        self.num_parallel_samples = num_parallel_samples

        # Regression
        self.num_targets = num_targets
        self.output_range = output_range

        super().__init__(**kwargs)

    def _num_patches(self):
        return (max(self.context_length, self.patch_length) - self.patch_length) // self.patch_stride + 1
