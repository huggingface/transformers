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

import math
from dataclasses import field

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ...utils.type_validators import activation_fn_key, interval, positive_int_field


@auto_docstring(checkpoint="eustlb/pocket-tts-audio-tokenizer")
@strict
class PocketTTSAudioTokenizerEncoderConfig(PreTrainedConfig):
    r"""
    latent_stride (`int`, *optional*, defaults to 16):
        Number of SEANet frames per latent frame: the stride of the encoder's convolution that downsamples the SEANet
        output to the latents, and of the decoder's transposed one that upsamples them back. A latent frame covers
        `prod(upsampling_ratios) * latent_stride` audio samples (`frame_size`, 1920 for the released checkpoints).
    latent_dim (`int`, *optional*, defaults to 32):
        Dimension of the continuous latents. Mimi's residual vector quantizer is replaced by this bottleneck: the
        encoder's downsampling convolution outputs `latent_dim` channels, and the decoder projects them back to
        `hidden_size` before upsampling.
    audio_channels (`int`, *optional*, defaults to 1):
        Number of channels in the audio data; the released checkpoints, and this implementation, are mono.
    num_filters (`int`, *optional*, defaults to 64):
        Number of convolution kernels of the first `PocketTTSAudioTokenizerConv1d` layer of the encoder.
    num_residual_layers (`int`, *optional*, defaults to 1):
        Number of residual layers.
    upsampling_ratios (`list[int]`, *optional*):
        Kernel size and stride ratios of the decoder's transposed convolutions; the encoder uses them in reverse order.
        If not specified, defaults to `[6, 5, 4]`.
    last_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size for the last convolution layer.
    residual_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size for the residual layers.
    dilation_growth_rate (`int`, *optional*, defaults to 2):
        How much to increase the dilation with each layer.
    use_causal_conv (`bool`, *optional*, defaults to `True`):
        Whether to use fully causal convolution.
    pad_mode (`str`, *optional*, defaults to `"constant"`):
        Padding mode for the convolutions.
    compress (`int`, *optional*, defaults to 2):
        Reduced dimensionality in residual branches.
    trim_right_ratio (`float`, *optional*, defaults to 1.0):
        Ratio for trimming at the right of the transposed convolution under the `use_causal_conv = True` setup. If
        equal to 1.0, it means that all the trimming is done at the right.
    upsample_groups (`int`, *optional*, defaults to 512):
        Number of groups of the transposed convolution that upsamples the latents to the SEANet rate.
    norm_eps (`float`, *optional*, defaults to 1e-05):
        The epsilon used by the layer normalization layers of the transformers.
    sliding_window (`int`, *optional*, defaults to 250):
        Sliding window of the transformers, in keys (the current frame included).
    layer_scale_initial_scale (`float`, *optional*, defaults to 0.01):
        Initial scale of the residual rescaling operation done in the transformer layers.
    """

    model_type = "pocket_tts_audio_tokenizer_encoder"
    base_config_key = "encoder_config"

    latent_stride: int = positive_int_field(default=16)
    latent_dim: int = positive_int_field(default=32)
    audio_channels: int = interval(min=1, max=1)(default=1)
    hidden_size: int = positive_int_field(default=512)
    num_filters: int = positive_int_field(default=64)
    num_residual_layers: int = positive_int_field(default=1)
    upsampling_ratios: list[int] | None = None
    kernel_size: int = 7
    last_kernel_size: int = 3
    residual_kernel_size: int = 3
    dilation_growth_rate: int = 2
    use_causal_conv: bool = True
    pad_mode: str = "constant"
    compress: int = 2
    trim_right_ratio: float = 1.0
    upsample_groups: int = positive_int_field(default=512)
    num_hidden_layers: int = positive_int_field(default=2)
    intermediate_size: int = positive_int_field(default=2048)
    num_attention_heads: int = positive_int_field(default=8)
    num_key_value_heads: int = positive_int_field(default=8)
    head_dim: int | None = None
    hidden_act: str = activation_fn_key(default="gelu_pytorch_tanh")
    max_position_embeddings: int = 8000
    initializer_range: float = interval(min=0.0, max=1.0)(default=0.02)
    norm_eps: float = interval(min=0.0)(default=1e-5)
    use_cache: bool = False
    rope_parameters: RopeParameters | dict | None = None
    sliding_window: int = positive_int_field(default=250)
    attention_dropout: float | int = interval(min=0.0, max=1.0)(default=0.0)
    layer_scale_initial_scale: float = interval(min=0.0)(default=0.01)
    attention_bias: bool = False

    def __post_init__(self, **kwargs):
        self.upsampling_ratios = self.upsampling_ratios if self.upsampling_ratios else [6, 5, 4]
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)

    @property
    def frame_size(self) -> int:
        """Number of audio samples per latent frame."""
        return math.prod(self.upsampling_ratios) * self.latent_stride


@auto_docstring(checkpoint="eustlb/pocket-tts-audio-tokenizer")
@strict
class PocketTTSAudioTokenizerDecoderConfig(PocketTTSAudioTokenizerEncoderConfig):
    r"""
    latent_stride (`int`, *optional*, defaults to 16):
        Number of SEANet frames per latent frame: the stride of the encoder's convolution that downsamples the SEANet
        output to the latents, and of the decoder's transposed one that upsamples them back. A latent frame covers
        `prod(upsampling_ratios) * latent_stride` audio samples (`frame_size`, 1920 for the released checkpoints).
    latent_dim (`int`, *optional*, defaults to 32):
        Dimension of the continuous latents. Mimi's residual vector quantizer is replaced by this bottleneck: the
        encoder's downsampling convolution outputs `latent_dim` channels, and the decoder projects them back to
        `hidden_size` before upsampling.
    audio_channels (`int`, *optional*, defaults to 1):
        Number of channels in the audio data; the released checkpoints, and this implementation, are mono.
    num_filters (`int`, *optional*, defaults to 64):
        Number of convolution kernels of the first `PocketTTSAudioTokenizerConv1d` layer of the encoder.
    num_residual_layers (`int`, *optional*, defaults to 1):
        Number of residual layers.
    upsampling_ratios (`list[int]`, *optional*):
        Kernel size and stride ratios of the decoder's transposed convolutions; the encoder uses them in reverse order.
        If not specified, defaults to `[6, 5, 4]`.
    last_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size for the last convolution layer.
    residual_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size for the residual layers.
    dilation_growth_rate (`int`, *optional*, defaults to 2):
        How much to increase the dilation with each layer.
    use_causal_conv (`bool`, *optional*, defaults to `True`):
        Whether to use fully causal convolution.
    pad_mode (`str`, *optional*, defaults to `"constant"`):
        Padding mode for the convolutions.
    compress (`int`, *optional*, defaults to 2):
        Reduced dimensionality in residual branches.
    trim_right_ratio (`float`, *optional*, defaults to 1.0):
        Ratio for trimming at the right of the transposed convolution under the `use_causal_conv = True` setup. If
        equal to 1.0, it means that all the trimming is done at the right.
    upsample_groups (`int`, *optional*, defaults to 512):
        Number of groups of the transposed convolution that upsamples the latents to the SEANet rate.
    sliding_window (`int`, *optional*, defaults to 250):
        Sliding window of the transformers, in keys (the current frame included).
    layer_scale_initial_scale (`float`, *optional*, defaults to 0.01):
        Initial scale of the residual rescaling operation done in the transformer layers.
    norm_eps (`float`, *optional*, defaults to 1e-05):
        The epsilon used by the layer normalization layers of the transformers.
    """

    model_type = "pocket_tts_audio_tokenizer_decoder"
    base_config_key = "decoder_config"


@auto_docstring(checkpoint="eustlb/pocket-tts-audio-tokenizer")
@strict
class PocketTTSAudioTokenizerConfig(PreTrainedConfig):
    r"""
    encoder_config (`Union[PocketTTSAudioTokenizerEncoderConfig, dict]`, *optional*):
        The config of the encoder, which turns audio into latents. `None` for a checkpoint without an encoder (or to load
        a full checkpoint's decoder only): the model then has no `encoder` and cannot `encode`.
    decoder_config (`Union[PocketTTSAudioTokenizerDecoderConfig, dict]`, *optional*):
        The config of the decoder, which turns latents into audio. `None` for a checkpoint without a decoder (or to load
        a full checkpoint's encoder only): the model then has no `decoder` and cannot `decode`.

    Example:

    ```python
    >>> from transformers import PocketTTSAudioTokenizerConfig, PocketTTSAudioTokenizerModel

    >>> # a full codec, and its decoder alone
    >>> model = PocketTTSAudioTokenizerModel(PocketTTSAudioTokenizerConfig())
    >>> decoder_only = PocketTTSAudioTokenizerModel(PocketTTSAudioTokenizerConfig(encoder_config=None))
    ```
    """

    model_type = "pocket_tts_audio_tokenizer"
    sub_configs_defaults = {
        "encoder_config": SubConfigSpec(config_class=PocketTTSAudioTokenizerEncoderConfig, optional=True),
        "decoder_config": SubConfigSpec(config_class=PocketTTSAudioTokenizerDecoderConfig, optional=True),
    }

    # present unless explicitly set to `None`
    encoder_config: PocketTTSAudioTokenizerEncoderConfig | dict | None = field(default_factory=dict)
    decoder_config: PocketTTSAudioTokenizerDecoderConfig | dict | None = field(default_factory=dict)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.encoder_config is None and self.decoder_config is None:
            raise ValueError("The audio tokenizer needs an encoder, a decoder, or both.")
        if self.encoder_config is not None and self.decoder_config is not None:
            for key in ("latent_dim", "frame_size"):
                if getattr(self.encoder_config, key) != getattr(self.decoder_config, key):
                    raise ValueError(f"The encoder and the decoder must share their `{key}`.")

    @property
    def latent_dim(self) -> int:
        """Dimension of the latents, shared by the encoder and the decoder."""
        return (self.decoder_config or self.encoder_config).latent_dim

    @property
    def frame_size(self) -> int:
        """Number of audio samples per latent frame, shared by the encoder and the decoder."""
        return (self.decoder_config or self.encoder_config).frame_size


__all__ = [
    "PocketTTSAudioTokenizerConfig",
    "PocketTTSAudioTokenizerEncoderConfig",
    "PocketTTSAudioTokenizerDecoderConfig",
]
