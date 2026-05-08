# Copyright 2023 The Kakao Enterprise Authors and the HuggingFace Inc. team. All rights reserved.
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
"""VITS model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/mms-tts-eng")
@strict
class VitsConfig(PreTrainedConfig):
    r"""
    window_size (`int`, *optional*, defaults to 4):
        Window size for the relative positional embeddings in the attention layers of the Transformer encoder.
    use_bias (`bool`, *optional*, defaults to `True`):
        Whether to use bias in the key, query, value projection layers in the Transformer encoder.
    ffn_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size of the 1D convolution layers used by the feed-forward network in the Transformer encoder.
    flow_size (`int`, *optional*, defaults to 192):
        Dimensionality of the flow layers.
    spectrogram_bins (`int`, *optional*, defaults to 513):
        Number of frequency bins in the target spectrogram.
    use_stochastic_duration_prediction (`bool`, *optional*, defaults to `True`):
        Whether to use the stochastic duration prediction module or the regular duration predictor.
    num_speakers (`int`, *optional*, defaults to 1):
        Number of speakers if this is a multi-speaker model.
    speaker_embedding_size (`int`, *optional*, defaults to 0):
        Number of channels used by the speaker embeddings. Is zero for single-speaker models.
    upsample_initial_channel (`int`, *optional*, defaults to 512):
        The number of input channels into the HiFi-GAN upsampling network.
    upsample_rates (`tuple[int]` or `list[int]`, *optional*, defaults to `[8, 8, 2, 2]`):
        A tuple of integers defining the stride of each 1D convolutional layer in the HiFi-GAN upsampling network.
        The length of `upsample_rates` defines the number of convolutional layers and has to match the length of
        `upsample_kernel_sizes`.
    upsample_kernel_sizes (`tuple[int]` or `list[int]`, *optional*, defaults to `[16, 16, 4, 4]`):
        A tuple of integers defining the kernel size of each 1D convolutional layer in the HiFi-GAN upsampling
        network. The length of `upsample_kernel_sizes` defines the number of convolutional layers and has to match
        the length of `upsample_rates`.
    resblock_kernel_sizes (`tuple[int]` or `list[int]`, *optional*, defaults to `[3, 7, 11]`):
        A tuple of integers defining the kernel sizes of the 1D convolutional layers in the HiFi-GAN
        multi-receptive field fusion (MRF) module.
    resblock_dilation_sizes (`tuple[tuple[int]]` or `list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
        A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
        HiFi-GAN multi-receptive field fusion (MRF) module.
    leaky_relu_slope (`float`, *optional*, defaults to 0.1):
        The angle of the negative slope used by the leaky ReLU activation.
    depth_separable_channels (`int`, *optional*, defaults to 2):
        Number of channels to use in each depth-separable block.
    depth_separable_num_layers (`int`, *optional*, defaults to 3):
        Number of convolutional layers to use in each depth-separable block.
    duration_predictor_flow_bins (`int`, *optional*, defaults to 10):
        Number of channels to map using the unonstrained rational spline in the duration predictor model.
    duration_predictor_tail_bound (`float`, *optional*, defaults to 5.0):
        Value of the tail bin boundary when computing the unconstrained rational spline in the duration predictor
        model.
    duration_predictor_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size of the 1D convolution layers used in the duration predictor model.
    duration_predictor_dropout (`float`, *optional*, defaults to 0.5):
        The dropout ratio for the duration predictor model.
    duration_predictor_num_flows (`int`, *optional*, defaults to 4):
        Number of flow stages used by the duration predictor model.
    duration_predictor_filter_channels (`int`, *optional*, defaults to 256):
        Number of channels for the convolution layers used in the duration predictor model.
    prior_encoder_num_flows (`int`, *optional*, defaults to 4):
        Number of flow stages used by the prior encoder flow model.
    prior_encoder_num_wavenet_layers (`int`, *optional*, defaults to 4):
        Number of WaveNet layers used by the prior encoder flow model.
    posterior_encoder_num_wavenet_layers (`int`, *optional*, defaults to 16):
        Number of WaveNet layers used by the posterior encoder model.
    wavenet_kernel_size (`int`, *optional*, defaults to 5):
        Kernel size of the 1D convolution layers used in the WaveNet model.
    wavenet_dilation_rate (`int`, *optional*, defaults to 1):
        Dilation rates of the dilated 1D convolutional layers used in the WaveNet model.
    wavenet_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the WaveNet layers.
    speaking_rate (`float`, *optional*, defaults to 1.0):
        Speaking rate. Larger values give faster synthesised speech.
    noise_scale (`float`, *optional*, defaults to 0.667):
        How random the speech prediction is. Larger values create more variation in the predicted speech.
    noise_scale_duration (`float`, *optional*, defaults to 0.8):
        How random the duration prediction is. Larger values create more variation in the predicted durations.

    Example:

    ```python
    >>> from transformers import VitsModel, VitsConfig

    >>> # Initializing a "facebook/mms-tts-eng" style configuration
    >>> configuration = VitsConfig()

    >>> # Initializing a model (with random weights) from the "facebook/mms-tts-eng" style configuration
    >>> model = VitsModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vits"

    vocab_size: int = 38
    hidden_size: int = 192
    num_hidden_layers: int = 6
    num_attention_heads: int = 2
    window_size: int = 4
    use_bias: bool = True
    ffn_dim: int = 768
    layerdrop: float | int = 0.1
    ffn_kernel_size: int = 3
    flow_size: int = 192
    spectrogram_bins: int = 513
    hidden_act: str = "relu"
    hidden_dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation_dropout: float | int = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_stochastic_duration_prediction: bool = True
    num_speakers: int = 1
    speaker_embedding_size: int = 0
    upsample_initial_channel: int = 512
    upsample_rates: list[int] | tuple[int, ...] = (8, 8, 2, 2)
    upsample_kernel_sizes: list[int] | tuple[int, ...] = (16, 16, 4, 4)
    resblock_kernel_sizes: list[int] | tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: list | tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    leaky_relu_slope: float = 0.1
    depth_separable_channels: int = 2
    depth_separable_num_layers: int = 3
    duration_predictor_flow_bins: int = 10
    duration_predictor_tail_bound: float = 5.0
    duration_predictor_kernel_size: int = 3
    duration_predictor_dropout: float | int = 0.5
    duration_predictor_num_flows: int = 4
    duration_predictor_filter_channels: int = 256
    prior_encoder_num_flows: int = 4
    prior_encoder_num_wavenet_layers: int = 4
    posterior_encoder_num_wavenet_layers: int = 16
    wavenet_kernel_size: int = 5
    wavenet_dilation_rate: int = 1
    wavenet_dropout: float | int = 0.0
    speaking_rate: float | int = 1.0
    noise_scale: float = 0.667
    noise_scale_duration: float = 0.8
    sampling_rate: int = 16_000
    pad_token_id: int | None = None

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if len(self.upsample_kernel_sizes) != len(self.upsample_rates):
            raise ValueError(
                f"The length of `upsample_kernel_sizes` ({len(self.upsample_kernel_sizes)}) must match the length of "
                f"`upsample_rates` ({len(self.upsample_rates)})"
            )


__all__ = ["VitsConfig"]
