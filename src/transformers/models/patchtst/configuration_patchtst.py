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

from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.utils import auto_docstring


@auto_docstring(checkpoint="ibm-granite/granite-timeseries-patchtst")
@strict
class PatchTSTConfig(PreTrainedConfig):
    r"""
    context_length (`int`, *optional*, defaults to 32):
        The context length of the input sequence.
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
        Define the stride of the patchification process.
    num_attention_heads (`int`, *optional*, defaults to 4):
        Number of attention heads for each attention layer in the Transformer encoder.
    share_embedding (`bool`, *optional*, defaults to `True`):
        Sharing the input embedding across all channels.
    channel_attention (`bool`, *optional*, defaults to `False`):
        Activate channel attention block in the Transformer to allow channels to attend each other.
    ffn_dim (`int`, *optional*, defaults to 512):
        Dimension of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    norm_type (`str` , *optional*, defaults to `"batchnorm"`):
        Normalization at each Transformer layer. Can be `"batchnorm"` or `"layernorm"`.
    norm_eps (`float`, *optional*, defaults to 1e-05):
        A value added to the denominator for numerical stability of normalization.
    positional_dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability in the positional embedding layer.
    path_dropout (`float`, *optional*, defaults to 0.0):
        The dropout path in the residual block.
    ff_dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability used between the two layers of the feed-forward networks.
    bias (`bool`, *optional*, defaults to `True`):
        Whether to add bias in the feed-forward networks.
    activation_function (`str`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (string) in the Transformer.`"gelu"` and `"relu"` are supported.
    pre_norm (`bool`, *optional*, defaults to `True`):
        Normalization is applied before self-attention if pre_norm is set to `True`. Otherwise, normalization is
        applied after residual block.
    positional_encoding_type (`str`, *optional*, defaults to `"sincos"`):
        Positional encodings. Options `"random"` and `"sincos"` are supported.
    use_cls_token (`bool`, *optional*, defaults to `False`):
        Whether cls token is used.
    init_std (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated normal weight initialization distribution.
    share_projection (`bool`, *optional*, defaults to `True`):
        Sharing the projection layer across different channels in the forecast head.
    scaling (`Union`, *optional*, defaults to `"std"`):
        Whether to scale the input targets via "mean" scaler, "std" scaler or no scaler if `None`. If `True`, the
        scaler is set to "mean".
    do_mask_input (`bool`, *optional*):
        Apply masking during the pretraining.
    mask_type (`str`, *optional*, defaults to `"random"`):
        Masking type. Only `"random"` and `"forecast"` are currently supported.
    random_mask_ratio (`float`, *optional*, defaults to 0.5):
        Masking ratio applied to mask the input data during random pretraining.
    num_forecast_mask_patches (`int` or `list`, *optional*, defaults to `[2]`):
        Number of patches to be masked at the end of each batch sample. If it is an integer,
        all the samples in the batch will have the same number of masked patches. If it is a list,
        samples in the batch will be randomly masked by numbers defined in the list. This argument is only used
        for forecast pretraining.
    channel_consistent_masking (`bool`, *optional*, defaults to `False`):
        If channel consistent masking is True, all the channels will have the same masking pattern.
    unmasked_channel_indices (`list`, *optional*):
        Indices of channels that are not masked during pretraining. Values in the list are number between 1 and
        `num_input_channels`
    mask_value (`int`, *optional*, defaults to 0):
        Values in the masked patches will be filled by `mask_value`.
    pooling_type (`str`, *optional*, defaults to `"mean"`):
        Pooling of the embedding. `"mean"`, `"max"` and `None` are supported.
    head_dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability for head.
    prediction_length (`int`, *optional*, defaults to 24):
        The prediction horizon that the model will output.
    num_targets (`int`, *optional*, defaults to 1):
        Number of targets for regression and classification tasks. For classification, it is the number of
        classes.
    output_range (`list`, *optional*):
        Output range for regression task. The range of output values can be set to enforce the model to produce
        values within a range.
    num_parallel_samples (`int`, *optional*, defaults to 100):
        The number of samples is generated in parallel for probabilistic prediction.

    ```python
    >>> from transformers import PatchTSTConfig, PatchTSTModel

    >>> # Initializing an PatchTST configuration with 12 time steps for prediction
    >>> configuration = PatchTSTConfig(prediction_length=12)

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = PatchTSTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "patchtst"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_attention_heads",
        "num_hidden_layers": "num_hidden_layers",
    }

    num_input_channels: int = 1
    context_length: int = 32
    distribution_output: str = "student_t"
    loss: str | None = "mse"
    patch_length: int = 1
    patch_stride: int = 1
    num_hidden_layers: int = 3
    d_model: int = 128
    num_attention_heads: int = 4
    share_embedding: bool = True
    channel_attention: bool = False
    ffn_dim: int = 512
    norm_type: str = "batchnorm"
    norm_eps: float = 1e-05
    attention_dropout: float | int = 0.0
    positional_dropout: float | int = 0.0
    path_dropout: float | int = 0.0
    ff_dropout: float | int = 0.0
    bias: bool = True
    activation_function: str = "gelu"
    pre_norm: bool = True
    positional_encoding_type: str = "sincos"
    use_cls_token: bool = False
    init_std: float = 0.02
    share_projection: bool = True
    scaling: str | bool | None = "std"
    do_mask_input: bool | None = None
    mask_type: str = "random"
    random_mask_ratio: float = 0.5
    num_forecast_mask_patches: list[int] | tuple[int, ...] | int | None = (2,)
    channel_consistent_masking: bool | None = False
    unmasked_channel_indices: list[int] | None = None
    mask_value: int = 0
    pooling_type: str | None = "mean"
    head_dropout: float | int = 0.0
    prediction_length: int = 24
    num_targets: int = 1
    output_range: list | None = None
    num_parallel_samples: int = 100


__all__ = ["PatchTSTConfig"]
