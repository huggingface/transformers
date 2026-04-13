# Copyright 2021 ASAPP Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""SEW model configuration"""

import functools
import operator

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="BAAI/seggpt-vit-large")
@strict
class SEWConfig(PreTrainedConfig):
    r"""
    squeeze_factor (`int`, *optional*, defaults to 2):
        Sequence length downsampling factor after the encoder and upsampling factor after the transformer.
    feat_proj_dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability for output of the feature encoder.
    final_dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for the final projection layer of [`SEWForCTC`].
    feat_extract_norm (`str`, *optional*, defaults to `"group"`):
        The norm to be applied to 1D convolutional layers in feature encoder. One of `"group"` for group
        normalization of only the first 1D convolutional layer or `"layer"` for layer normalization of all 1D
        convolutional layers.
    feat_extract_activation (`str, `optional`, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the 1D convolutional layers of the feature
        extractor. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
    conv_dim (`tuple[int]` or `list[int]`, *optional*, defaults to `(64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512)`):
        A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
        feature encoder. The length of *conv_dim* defines the number of 1D convolutional layers.
    conv_stride (`tuple[int]` or `list[int]`, *optional*, defaults to `(5, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1)`):
        A tuple of integers defining the stride of each 1D convolutional layer in the feature encoder. The length
        of *conv_stride* defines the number of convolutional layers and has to match the length of *conv_dim*.
    conv_kernel (`tuple[int]` or `list[int]`, *optional*, defaults to `(10, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 2, 1)`):
        A tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder. The
        length of *conv_kernel* defines the number of convolutional layers and has to match the length of
        *conv_dim*.
    conv_bias (`bool`, *optional*, defaults to `False`):
        Whether the 1D convolutional layers have a bias.
    num_conv_pos_embeddings (`int`, *optional*, defaults to 128):
        Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
        embeddings layer.
    num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16):
        Number of groups of 1D convolutional positional embeddings layer.
    apply_spec_augment (`bool`, *optional*, defaults to `True`):
        Whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder. For reference see
        [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
        Recognition](https://huggingface.co/papers/1904.08779).
    mask_time_prob (`float`, *optional*, defaults to 0.05):
        Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
        procedure generates ''mask_time_prob*len(time_axis)/mask_time_length'' independent masks over the axis. If
        reasoning from the probability of each feature vector to be chosen as the start of the vector span to be
        masked, *mask_time_prob* should be `prob_vector_start*mask_time_length`. Note that overlap may decrease the
        actual percentage of masked vectors. This is only relevant if `apply_spec_augment is True`.
    mask_time_length (`int`, *optional*, defaults to 10):
        Length of vector span along the time axis.
    mask_time_min_masks (`int`, *optional*, defaults to 2),:
        The minimum number of masks of length `mask_feature_length` generated along the time axis, each time step,
        irrespectively of `mask_feature_prob`. Only relevant if ''mask_time_prob*len(time_axis)/mask_time_length <
        mask_time_min_masks''
    mask_feature_prob (`float`, *optional*, defaults to 0.0):
        Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
        masking procedure generates ''mask_feature_prob*len(feature_axis)/mask_time_length'' independent masks over
        the axis. If reasoning from the probability of each feature vector to be chosen as the start of the vector
        span to be masked, *mask_feature_prob* should be `prob_vector_start*mask_feature_length`. Note that overlap
        may decrease the actual percentage of masked vectors. This is only relevant if `apply_spec_augment is
        True`.
    mask_feature_length (`int`, *optional*, defaults to 10):
        Length of vector span along the feature axis.
    mask_feature_min_masks (`int`, *optional*, defaults to 0):
        The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
        step, irrespectively of `mask_feature_prob`. Only relevant if
        ''mask_feature_prob*len(feature_axis)/mask_feature_length < mask_feature_min_masks''
    ctc_zero_infinity (`bool`, *optional*, defaults to `False`):
        Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
        occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
        of [`SEWForCTC`].
    use_weighted_layer_sum (`bool`, *optional*, defaults to `False`):
        Whether to use a weighted average of layer outputs with learned weights. Only relevant when using an
        instance of [`Wav2Vec2ForSequenceClassification`].
    classifier_proj_size (`int`, *optional*, defaults to 256):
        Dimensionality of the projection before token mean-pooling for classification.

    Example:

    ```python
    >>> from transformers import SEWConfig, SEWModel

    >>> # Initializing a SEW asapp/sew-tiny-100k style configuration
    >>> configuration = SEWConfig()

    >>> # Initializing a model (with random weights) from the asapp/sew-tiny-100k style configuration
    >>> model = SEWModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "sew"

    vocab_size: int = 32
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    squeeze_factor: int = 2
    hidden_act: str = "gelu"
    hidden_dropout: float | int = 0.1
    activation_dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    feat_proj_dropout: float | int = 0.0
    final_dropout: float | int = 0.1
    layerdrop: float | int = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    feat_extract_norm: str = "group"
    feat_extract_activation: str = "gelu"
    conv_dim: list[int] | tuple[int, ...] = (64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512)
    conv_stride: list[int] | tuple[int, ...] = (5, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1)
    conv_kernel: list[int] | tuple[int, ...] = (10, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 2, 1)
    conv_bias: bool = False
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    apply_spec_augment: bool = True
    mask_time_prob: float | int = 0.05
    mask_time_length: int = 10
    mask_time_min_masks: int = 2
    mask_feature_prob: float | int = 0.0
    mask_feature_length: int = 10
    mask_feature_min_masks: int = 0
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = False
    use_weighted_layer_sum: bool = False
    classifier_proj_size: int = 256
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2

    def __post_init__(self, **kwargs):
        self.num_feat_extract_layers = len(self.conv_dim)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if (
            (len(self.conv_stride) != self.num_feat_extract_layers)
            or (len(self.conv_kernel) != self.num_feat_extract_layers)
            or (len(self.conv_dim) != self.num_feat_extract_layers)
        ):
            raise ValueError(
                "Configuration for convolutional layers is incorrect. "
                "It is required that `len(config.conv_dim)` == `len(config.conv_stride)` == `len(config.conv_kernel)`, "
                f"but is `len(config.conv_dim) = {len(self.conv_dim)}`, `len(config.conv_stride) "
                f"= {len(self.conv_stride)}`, `len(config.conv_kernel) = {len(self.conv_kernel)}`."
            )

    @property
    def inputs_to_logits_ratio(self):
        return functools.reduce(operator.mul, self.conv_stride, 1)


__all__ = ["SEWConfig"]
