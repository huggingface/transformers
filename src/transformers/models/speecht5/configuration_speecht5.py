# Copyright 2023 The Fairseq Authors, Microsoft Research, and the HuggingFace Inc. team. All rights reserved.
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
"""SpeechT5 model configuration"""

import functools
import operator

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/speecht5_asr")
@strict
class SpeechT5Config(PreTrainedConfig):
    r"""
    positional_dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for the text position encoding layers.
    feat_extract_norm (`str`, *optional*, defaults to `"group"`):
        The norm to be applied to 1D convolutional layers in the speech encoder pre-net. One of `"group"` for group
        normalization of only the first 1D convolutional layer or `"layer"` for layer normalization of all 1D
        convolutional layers.
    feat_proj_dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability for output of the speech encoder pre-net.
    feat_extract_activation (`str, `optional`, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the 1D convolutional layers of the feature
        extractor. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
    conv_dim (`tuple[int]` or `list[int]`, *optional*, defaults to `(512, 512, 512, 512, 512, 512, 512)`):
        A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
        speech encoder pre-net. The length of *conv_dim* defines the number of 1D convolutional layers.
    conv_stride (`tuple[int]` or `list[int]`, *optional*, defaults to `(5, 2, 2, 2, 2, 2, 2)`):
        A tuple of integers defining the stride of each 1D convolutional layer in the speech encoder pre-net. The
        length of *conv_stride* defines the number of convolutional layers and has to match the length of
        *conv_dim*.
    conv_kernel (`tuple[int]` or `list[int]`, *optional*, defaults to `(10, 3, 3, 3, 3, 3, 3)`):
        A tuple of integers defining the kernel size of each 1D convolutional layer in the speech encoder pre-net.
        The length of *conv_kernel* defines the number of convolutional layers and has to match the length of
        *conv_dim*.
    conv_bias (`bool`, *optional*, defaults to `False`):
        Whether the 1D convolutional layers have a bias.
    num_conv_pos_embeddings (`int`, *optional*, defaults to 128):
        Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
        embeddings layer.
    num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16):
        Number of groups of 1D convolutional positional embeddings layer.
    apply_spec_augment (`bool`, *optional*, defaults to `True`):
        Whether to apply *SpecAugment* data augmentation to the outputs of the speech encoder pre-net. For
        reference see [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
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
    mask_feature_min_masks (`int`, *optional*, defaults to 0),:
        The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
        step, irrespectively of `mask_feature_prob`. Only relevant if
        ''mask_feature_prob*len(feature_axis)/mask_feature_length < mask_feature_min_masks''
    num_mel_bins (`int`, *optional*, defaults to 80):
        Number of mel features used per input features. Used by the speech decoder pre-net. Should correspond to
        the value used in the [`SpeechT5Processor`] class.
    speech_decoder_prenet_layers (`int`, *optional*, defaults to 2):
        Number of layers in the speech decoder pre-net.
    speech_decoder_prenet_units (`int`, *optional*, defaults to 256):
        Dimensionality of the layers in the speech decoder pre-net.
    speech_decoder_prenet_dropout (`float`, *optional*, defaults to 0.5):
        The dropout probability for the speech decoder pre-net layers.
    speaker_embedding_dim (`int`, *optional*, defaults to 512):
        Dimensionality of the *XVector* embedding vectors.
    speech_decoder_postnet_layers (`int`, *optional*, defaults to 5):
        Number of layers in the speech decoder post-net.
    speech_decoder_postnet_units (`int`, *optional*, defaults to 256):
        Dimensionality of the layers in the speech decoder post-net.
    speech_decoder_postnet_kernel (`int`, *optional*, defaults to 5):
        Number of convolutional filter channels in the speech decoder post-net.
    speech_decoder_postnet_dropout (`float`, *optional*, defaults to 0.5):
        The dropout probability for the speech decoder post-net layers.
    reduction_factor (`int`, *optional*, defaults to 2):
        Spectrogram length reduction factor for the speech decoder inputs.
    max_speech_positions (`int`, *optional*, defaults to 4000):
        The maximum sequence length of speech features that this model might ever be used with.
    max_text_positions (`int`, *optional*, defaults to 450):
        The maximum sequence length of text features that this model might ever be used with.
    encoder_max_relative_position (`int`, *optional*, defaults to 160):
        Maximum distance for relative position embedding in the encoder.
    use_guided_attention_loss (`bool`, *optional*, defaults to `True`):
        Whether to apply guided attention loss while training the TTS model.
    guided_attention_loss_num_heads (`int`, *optional*, defaults to 2):
        Number of attention heads the guided attention loss will be applied to. Use -1 to apply this loss to all
        attention heads.
    guided_attention_loss_sigma (`float`, *optional*, defaults to 0.4):
        Standard deviation for guided attention loss.
    guided_attention_loss_scale (`float`, *optional*, defaults to 10.0):
        Scaling coefficient for guided attention loss (also known as lambda).

    Example:

    ```python
    >>> from transformers import SpeechT5Model, SpeechT5Config

    >>> # Initializing a "microsoft/speecht5_asr" style configuration
    >>> configuration = SpeechT5Config()

    >>> # Initializing a model (with random weights) from the "microsoft/speecht5_asr" style configuration
    >>> model = SpeechT5Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "speecht5"
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "num_hidden_layers": "encoder_layers"}

    vocab_size: int = 81
    hidden_size: int = 768
    encoder_layers: int = 12
    encoder_attention_heads: int = 12
    encoder_ffn_dim: int = 3072
    encoder_layerdrop: float | int = 0.1
    decoder_layers: int = 6
    decoder_ffn_dim: int = 3072
    decoder_attention_heads: int = 12
    decoder_layerdrop: float | int = 0.1
    hidden_act: str = "gelu"
    positional_dropout: float | int = 0.1
    hidden_dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation_dropout: float | int = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    scale_embedding: bool = False
    feat_extract_norm: str = "group"
    feat_proj_dropout: float | int = 0.0
    feat_extract_activation: str = "gelu"
    conv_dim: list[int] | tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    conv_stride: list[int] | tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_kernel: list[int] | tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_bias: bool = False
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    apply_spec_augment: bool = True
    mask_time_prob: float = 0.05
    mask_time_length: int = 10
    mask_time_min_masks: int = 2
    mask_feature_prob: float = 0.0
    mask_feature_length: int = 10
    mask_feature_min_masks: int = 0
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    decoder_start_token_id: int | None = 2
    num_mel_bins: int = 80
    speech_decoder_prenet_layers: int = 2
    speech_decoder_prenet_units: int = 256
    speech_decoder_prenet_dropout: float | int = 0.5
    speaker_embedding_dim: int = 512
    speech_decoder_postnet_layers: int = 5
    speech_decoder_postnet_units: int = 256
    speech_decoder_postnet_kernel: int = 5
    speech_decoder_postnet_dropout: float | int = 0.5
    reduction_factor: int = 2
    max_speech_positions: int = 4000
    max_text_positions: int = 450
    encoder_max_relative_position: int = 160
    use_guided_attention_loss: bool = True
    guided_attention_loss_num_heads: int = 2
    guided_attention_loss_sigma: float = 0.4
    guided_attention_loss_scale: float = 10.0
    use_cache: bool = True
    is_encoder_decoder: bool = True
    tie_word_embeddings: bool = True

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
                "Configuration for convolutional layers is incorrect. It is required that `len(config.conv_dim)` =="
                " `len(config.conv_stride)` == `len(config.conv_kernel)`, but is `len(config.conv_dim) ="
                f" {len(self.conv_dim)}`, `len(config.conv_stride) = {len(self.conv_stride)}`,"
                f" `len(config.conv_kernel) = {len(self.conv_kernel)}`."
            )

    def inputs_to_logits_ratio(self):
        return functools.reduce(operator.mul, self.conv_stride, 1)


@auto_docstring(checkpoint="microsoft/speecht5_asr")
@strict
class SpeechT5HifiGanConfig(PreTrainedConfig):
    r"""
    model_in_dim (`int`, *optional*, defaults to 80):
        The number of frequency bins in the input log-mel spectrogram.
    upsample_initial_channel (`int`, *optional*, defaults to 512):
        The number of input channels into the upsampling network.
    upsample_rates (`tuple[int]` or `list[int]`, *optional*, defaults to `[4, 4, 4, 4]`):
        A tuple of integers defining the stride of each 1D convolutional layer in the upsampling network. The
        length of *upsample_rates* defines the number of convolutional layers and has to match the length of
        *upsample_kernel_sizes*.
    upsample_kernel_sizes (`tuple[int]` or `list[int]`, *optional*, defaults to `[8, 8, 8, 8]`):
        A tuple of integers defining the kernel size of each 1D convolutional layer in the upsampling network. The
        length of *upsample_kernel_sizes* defines the number of convolutional layers and has to match the length of
        *upsample_rates*.
    resblock_kernel_sizes (`tuple[int]` or `list[int]`, *optional*, defaults to `[3, 7, 11]`):
        A tuple of integers defining the kernel sizes of the 1D convolutional layers in the multi-receptive field
        fusion (MRF) module.
    resblock_dilation_sizes (`tuple[tuple[int]]` or `list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
        A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
        multi-receptive field fusion (MRF) module.
    leaky_relu_slope (`float`, *optional*, defaults to 0.1):
        The angle of the negative slope used by the leaky ReLU activation.
    normalize_before (`bool`, *optional*, defaults to `True`):
        Whether or not to normalize the spectrogram before vocoding using the vocoder's learned mean and variance.

    Example:

    ```python
    >>> from transformers import SpeechT5HifiGan, SpeechT5HifiGanConfig

    >>> # Initializing a "microsoft/speecht5_hifigan" style configuration
    >>> configuration = SpeechT5HifiGanConfig()

    >>> # Initializing a model (with random weights) from the "microsoft/speecht5_hifigan" style configuration
    >>> model = SpeechT5HifiGan(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "hifigan"

    model_in_dim: int = 80
    sampling_rate: int = 16000
    upsample_initial_channel: int = 512
    upsample_rates: list[int] | tuple[int, ...] = (4, 4, 4, 4)
    upsample_kernel_sizes: list[int] | tuple[int, ...] = (8, 8, 8, 8)
    resblock_kernel_sizes: list[int] | tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: list | tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    initializer_range: float = 0.01
    leaky_relu_slope: float = 0.1
    normalize_before: bool = True


__all__ = ["SpeechT5Config", "SpeechT5HifiGanConfig"]
