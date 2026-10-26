# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Qwen3TTS Multi-Codebook Tokenizer model."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedAudioTokenizerBase
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..dac.modeling_dac import DacDecoderOutput
from ..mimi.configuration_mimi import MimiConfig
from ..mimi.modeling_mimi import (
    MimiConv1d,
    MimiEncoder,
    MimiEuclideanCodebook,
    MimiPreTrainedModel,
    MimiResidualVectorQuantizer,
    MimiSplitResidualVectorQuantizer,
    MimiTransformerModel,
    MimiVectorQuantization,
)
from ..qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniSnakeBeta
from ..qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeCode2WavConfig
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCausalConvNet,
    Qwen3OmniMoeCausalTransConvNet,
    Qwen3OmniMoeCode2WavAttention,
    Qwen3OmniMoeCode2WavDecoderBlock,
    Qwen3OmniMoeCode2WavTransformerLayer,
    Qwen3OmniMoeCode2WavTransformerModel,
    Qwen3OmniMoeConvNeXtBlock,
)
from ..vibevoice_acoustic_tokenizer.modeling_vibevoice_acoustic_tokenizer import VibeVoiceAcousticTokenizerModel


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Qwen/Qwen3-TTS-Tokenizer-12Hz")
@strict
class Qwen3TTSTokenizerSemanticQuantizerConfig(PreTrainedConfig):
    r"""
    vector_quantization_hidden_dimension (`int`, *optional*, defaults to 256):
        Dimension used within the semantic vector quantizers.
    """

    model_type = "qwen3_tts_tokenizer_semantic_quantizer"
    base_config_key = "semantic_quantizer_config"
    attribute_map = {"num_quantizers": "n_codebooks"}

    codebook_size: int = 2048
    codebook_dim: int = 256
    n_codebooks: int = 1
    vector_quantization_hidden_dimension: int = 256
    hidden_size: int = 512

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.n_codebooks < 1:
            raise ValueError(
                f"The semantic quantizer needs at least one codebook, but `n_codebooks` is {self.n_codebooks}."
            )


@auto_docstring(checkpoint="Qwen/Qwen3-TTS-Tokenizer-12Hz")
@strict
class Qwen3TTSTokenizerAcousticQuantizerConfig(PreTrainedConfig):
    r"""
    vector_quantization_hidden_dimension (`int`, *optional*, defaults to 256):
        Dimension used within the acoustic vector quantizers.
    """

    model_type = "qwen3_tts_tokenizer_acoustic_quantizer"
    base_config_key = "acoustic_quantizer_config"
    attribute_map = {"num_quantizers": "n_codebooks"}

    codebook_size: int = 2048
    codebook_dim: int = 256
    n_codebooks: int = 15
    vector_quantization_hidden_dimension: int = 256
    hidden_size: int = 512

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.n_codebooks < 0:
            raise ValueError(f"`n_codebooks` must not be negative, but is {self.n_codebooks}.")


@auto_docstring(checkpoint="Qwen/Qwen3-TTS-Tokenizer-12Hz")
@strict
class Qwen3TTSTokenizerEncoderConfig(MimiConfig):
    r"""
    audio_channels (`int`, *optional*, defaults to 1):
        Number of channels in the audio data. Either 1 for mono or 2 for stereo.
    num_filters (`int`, *optional*, defaults to 64):
        Number of convolution kernels of first `Qwen3TTSTokenizerConv1d` down sampling layer.
    num_residual_layers (`int`,  *optional*, defaults to 1):
        Number of residual layers.
    upsampling_ratios (`Sequence[int]`, *optional*):
        Kernel size and stride ratios. The encoder uses downsampling ratios instead of upsampling ratios, hence it
        will use the ratios in the reverse order to the ones specified here. If not specified, will defaults to
        `[8, 6, 5, 4]`.
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
    use_conv_shortcut (`bool`, *optional*, defaults to `False`):
        Whether to use a convolutional layer as the 'skip' connection in the `Qwen3TTSTokenizerResnetBlock` block. If
        False, an identity function will be used, giving a generic residual connection.
    """

    model_type = "qwen3_tts_tokenizer_encoder"
    base_config_key = "encoder_config"

    upsampling_ratios: list[int] | tuple[int, ...] | None = None

    codebook_size = AttributeError()
    codebook_dim = AttributeError()
    num_quantizers = AttributeError()
    num_semantic_quantizers = AttributeError()
    vector_quantization_hidden_dimension = AttributeError()
    trim_right_ratio = AttributeError()
    upsample_groups = AttributeError()
    use_cache = AttributeError()
    use_streaming = AttributeError()
    tie_word_embeddings = AttributeError()
    encoder_config = AttributeError()
    decoder_config = AttributeError()
    latent_stride = AttributeError()
    latent_dim = AttributeError()

    def __post_init__(self, **kwargs):
        self.upsampling_ratios = self.upsampling_ratios if self.upsampling_ratios else [8, 6, 5, 4]
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        self._frame_rate = kwargs.pop("frame_rate", None)
        PreTrainedConfig.__post_init__(self, **kwargs)

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get("model_type") == Qwen3TTSTokenizerConfig.model_type:
            config_dict = Qwen3TTSTokenizerConfig.from_dict(config_dict).encoder_config.to_dict()
        return config_dict, kwargs

    def validate_architecture(self):
        raise AttributeError("Not needed for Qwen3TTSTokenizer")

    def num_codebooks(self):
        raise AttributeError("Not needed for Qwen3TTSTokenizer")

    def to_dict(self):
        raise AttributeError("Use PreTrainedConfig serialization")


@auto_docstring(checkpoint="Qwen/Qwen3-TTS-Tokenizer-12Hz")
@strict
class Qwen3TTSTokenizerDecoderConfig(Qwen3OmniMoeCode2WavConfig):
    r"""
    upsample_rates (`Tuple[int]`, *optional*, defaults to `(8, 5, 4, 3)`):
        Rate at which features are upsampled in the final waveform synthesis stage.
    upsampling_ratios (`Tuple[int]`, *optional*, defaults to `(2, 2)`):
        Ratios used in transposed convolutional layers to progressively upsample feature maps to waveform.
    decoder_dim (`int`, *optional*, defaults to 1536):
        Final dimensionality of the decoder's output before waveform generation.
    codebook_dim (`int`, *optional*, defaults to 512):
        Width of the continuous representation passed to the decoder, not the quantizer's codebook embedding dimension.
    latent_dim (`int`, *optional*, defaults to 1024):
        Latent dimension used between pre-conv and transformer.
    decoder_chunk_size (`int`, *optional*, defaults to 300):
        Number of frames the decoder processes per chunk.
    decoder_left_context_size (`int`, *optional*, defaults to 25):
        Number of preceding frames prepended to each decoder chunk as context and dropped from its output.
    """

    model_type = "qwen3_tts_tokenizer_decoder"
    base_config_key = "decoder_config"

    hidden_size: int = 512
    intermediate_size: int = 1024
    head_dim: int = 64
    codebook_dim: int = 512
    latent_dim: int = 1024
    decoder_chunk_size: int = 300
    decoder_left_context_size: int = 25

    codebook_size = AttributeError()
    num_quantizers = AttributeError()

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get("model_type") == Qwen3TTSTokenizerConfig.model_type:
            config_dict = Qwen3TTSTokenizerConfig.from_dict(config_dict).decoder_config.to_dict()
        return config_dict, kwargs


@auto_docstring(checkpoint="Qwen/Qwen3-TTS-Tokenizer-12Hz")
@strict
class Qwen3TTSTokenizerConfig(PreTrainedConfig):
    r"""
    encoder_hidden_size (`int`, *optional*, defaults to 512):
        Hidden size of the encoder transformer. Encoder architecture settings use the `encoder_` prefix and are
        independent of the decoder settings stored without a prefix.
    encoder_audio_channels (`int`, *optional*, defaults to 1):
        Number of input audio channels.
    encoder_num_filters (`int`, *optional*, defaults to 64):
        Number of filters in the encoder's first convolution.
    encoder_num_residual_layers (`int`, *optional*, defaults to 1):
        Number of residual layers in each encoder convolutional stage.
    encoder_kernel_size (`int`, *optional*, defaults to 7):
        Kernel size of the encoder's first convolution.
    encoder_last_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size of the encoder's last convolution.
    encoder_residual_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size of encoder residual convolutions.
    encoder_dilation_growth_rate (`int`, *optional*, defaults to 2):
        Growth factor for encoder residual convolution dilations.
    encoder_use_causal_conv (`bool`, *optional*, defaults to `True`):
        Whether the encoder uses causal convolutions.
    encoder_pad_mode (`str`, *optional*, defaults to `"constant"`):
        Padding mode of encoder convolutions.
    encoder_compress (`int`, *optional*, defaults to 2):
        Channel compression factor in encoder residual branches.
    encoder_use_conv_shortcut (`bool`, *optional*, defaults to `False`):
        Whether encoder residual shortcuts use a convolution instead of an identity.
    encoder_num_hidden_layers (`int`, *optional*, defaults to 8):
        Number of encoder transformer layers.
    encoder_intermediate_size (`int`, *optional*, defaults to 2048):
        Intermediate size of encoder transformer feed-forward layers.
    encoder_num_attention_heads (`int`, *optional*, defaults to 8):
        Number of encoder attention query heads.
    encoder_num_key_value_heads (`int`, *optional*, defaults to 8):
        Number of encoder attention key and value heads.
    encoder_head_dim (`int`, *optional*):
        Encoder attention head dimension. Defaults to `encoder_hidden_size // encoder_num_attention_heads`.
    encoder_hidden_act (`str`, *optional*, defaults to `"gelu"`):
        Activation function of encoder transformer feed-forward layers.
    encoder_max_position_embeddings (`int`, *optional*, defaults to 8000):
        Maximum sequence length of the encoder transformer.
    encoder_initializer_range (`float`, *optional*, defaults to 0.02):
        Standard deviation used to initialize encoder weights.
    encoder_norm_eps (`float`, *optional*, defaults to 1e-5):
        Epsilon of encoder layer normalization.
    encoder_rope_parameters (`dict`, *optional*):
        Rotary position embedding parameters of the encoder transformer.
    encoder_sliding_window (`int`, *optional*, defaults to 250):
        Sliding attention window of the encoder transformer.
    encoder_attention_dropout (`float`, *optional*, defaults to 0.0):
        Dropout probability of encoder attention weights.
    encoder_layer_scale_initial_scale (`float`, *optional*, defaults to 0.01):
        Initial value of encoder transformer layer scales.
    encoder_attention_bias (`bool`, *optional*, defaults to `False`):
        Whether encoder attention projections use biases.
    encoder_upsampling_ratios (`list[int]` or `tuple[int, ...]`, *optional*, defaults to `(8, 6, 5, 4)`):
        Convolutional downsampling ratios used by the encoder, in reverse order.
    encoder_frame_rate (`float` or `int`, *optional*):
        Encoder frame rate. If unset, derived from the input sampling rate and encoder convolution strides.
    acoustic_quantizer_config (`dict` or `Qwen3TTSTokenizerAcousticQuantizerConfig`, *optional*):
        Configuration for the acoustic residual vector quantizer.
    semantic_quantizer_config (`dict` or `Qwen3TTSTokenizerSemanticQuantizerConfig`, *optional*):
        Configuration for the semantic residual vector quantizer, applied before the acoustic one.
    upsample_rates (`Tuple[int]`, *optional*, defaults to `(8, 5, 4, 3)`):
        Rate at which features are upsampled in the final waveform synthesis stage.
    upsampling_ratios (`Tuple[int]`, *optional*, defaults to `(2, 2)`):
        Ratios used in transposed convolutional layers to progressively upsample feature maps to waveform.
    decoder_dim (`int`, *optional*, defaults to 1536):
        Final dimensionality of the decoder's output before waveform generation.
    codebook_dim (`int`, *optional*, defaults to 512):
        Width of the continuous representation passed to the decoder, not the quantizers' `codebook_dim`.
    latent_dim (`int`, *optional*, defaults to 1024):
        Latent dimension used between pre-conv and transformer.
    decoder_chunk_size (`int`, *optional*, defaults to 300):
        Number of frames the decoder processes per chunk.
    decoder_left_context_size (`int`, *optional*, defaults to 25):
        Number of preceding frames prepended to each decoder chunk as context and dropped from its output.
    input_sampling_rate (`int`, *optional*, defaults to 24000):
        Sampling rate, in hertz (Hz), of the encoder's input audio waveform.
    output_sampling_rate (`int`, *optional*, defaults to 24000):
        Sampling rate, in hertz (Hz), of the decoder's output audio waveform.
    """

    model_type = "qwen3_tts_tokenizer_12hz"
    base_config_key = ""
    attribute_map = {
        "input_sample_rate": "input_sampling_rate",
        "output_sample_rate": "output_sampling_rate",
    }
    sub_configs = {
        "acoustic_quantizer_config": Qwen3TTSTokenizerAcousticQuantizerConfig,
        "semantic_quantizer_config": Qwen3TTSTokenizerSemanticQuantizerConfig,
    }

    encoder_audio_channels: int = 1
    encoder_hidden_size: int = 512
    encoder_num_filters: int = 64
    encoder_num_residual_layers: int = 1
    encoder_upsampling_ratios: list[int] | tuple[int, ...] = (8, 6, 5, 4)
    encoder_kernel_size: int = 7
    encoder_last_kernel_size: int = 3
    encoder_residual_kernel_size: int = 3
    encoder_dilation_growth_rate: int = 2
    encoder_use_causal_conv: bool = True
    encoder_pad_mode: str = "constant"
    encoder_compress: int = 2
    encoder_use_conv_shortcut: bool = False
    encoder_num_hidden_layers: int = 8
    encoder_intermediate_size: int = 2048
    encoder_num_attention_heads: int = 8
    encoder_num_key_value_heads: int = 8
    encoder_head_dim: int | None = None
    encoder_hidden_act: str = "gelu"
    encoder_max_position_embeddings: int = 8000
    encoder_initializer_range: float = 0.02
    encoder_norm_eps: float = 1e-5
    encoder_rope_parameters: dict | None = None
    encoder_sliding_window: int = 250
    encoder_attention_dropout: float | int = 0.0
    encoder_layer_scale_initial_scale: float = 0.01
    encoder_attention_bias: bool = False
    encoder_frame_rate: float | int | None = None

    hidden_size: int = 512
    max_position_embeddings: int = 8000
    rope_parameters: dict | None = None
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    attention_bias: bool = False
    sliding_window: int = 72
    intermediate_size: int = 1024
    hidden_act: str = "silu"
    layer_scale_initial_scale: float = 0.01
    rms_norm_eps: float = 1e-5
    num_hidden_layers: int = 8
    upsample_rates: list[int] | tuple[int, ...] = (8, 5, 4, 3)
    upsampling_ratios: list[int] | tuple[int, ...] = (2, 2)
    decoder_dim: int = 1536
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    head_dim: int = 64
    codebook_dim: int = 512
    latent_dim: int = 1024
    decoder_chunk_size: int = 300
    decoder_left_context_size: int = 25

    acoustic_quantizer_config: dict | PreTrainedConfig | None = None
    semantic_quantizer_config: dict | PreTrainedConfig | None = None
    input_sampling_rate: int | None = 24000
    output_sampling_rate: int | None = 24000

    def __post_init__(self, **kwargs):
        if self.acoustic_quantizer_config is None:
            self.acoustic_quantizer_config = Qwen3TTSTokenizerAcousticQuantizerConfig()
        elif isinstance(self.acoustic_quantizer_config, dict):
            self.acoustic_quantizer_config = Qwen3TTSTokenizerAcousticQuantizerConfig(**self.acoustic_quantizer_config)
        if self.semantic_quantizer_config is None:
            self.semantic_quantizer_config = Qwen3TTSTokenizerSemanticQuantizerConfig()
        elif isinstance(self.semantic_quantizer_config, dict):
            self.semantic_quantizer_config = Qwen3TTSTokenizerSemanticQuantizerConfig(**self.semantic_quantizer_config)
        super().__post_init__(**kwargs)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # Published checkpoints store architecture and unused quantizer metadata in nested dictionaries.
        # Normalize once on input; constructors and saved configs use only the canonical flat fields.
        config_dict = deepcopy(config_dict)
        encoder_config = config_dict.pop("encoder_config", None) or {}
        decoder_config = config_dict.pop("decoder_config", None) or {}
        config_dict.pop("encoder_valid_num_quantizers", None)
        for old_key, new_key in (
            ("input_sample_rate", "input_sampling_rate"),
            ("output_sample_rate", "output_sampling_rate"),
        ):
            if old_key in config_dict:
                config_dict.setdefault(new_key, config_dict.pop(old_key))
        for key, value in encoder_config.items():
            canonical_key = {
                "sampling_rate": "input_sampling_rate",
                "_frame_rate": "encoder_frame_rate",
            }.get(key, f"encoder_{key}")
            if canonical_key in cls.__dataclass_fields__:
                config_dict.setdefault(canonical_key, value)
        if "encoder_rope_parameters" not in config_dict and (
            "rope_theta" in encoder_config or "rope_scaling" in encoder_config
        ):
            rope_parameters = deepcopy(encoder_config.get("rope_scaling") or {})
            rope_parameters.setdefault("rope_type", rope_parameters.pop("type", "default"))
            rope_parameters.setdefault("rope_theta", encoder_config.get("rope_theta", 10000.0))
            config_dict["encoder_rope_parameters"] = rope_parameters
        for key, value in decoder_config.items():
            if key in Qwen3TTSTokenizerDecoderConfig.__annotations__ or key in ("rope_theta", "rope_scaling"):
                config_dict.setdefault(key, value)
        quantizer_keys = ("acoustic_quantizer_config", "semantic_quantizer_config")
        for key in tuple(kwargs):
            if key in cls.__dataclass_fields__ and key not in quantizer_keys:
                config_dict[key] = deepcopy(kwargs.pop(key))
        if decoder_config:
            # The original checkpoint describes both quantizers with shared settings in its decoder config.
            input_width = config_dict.get("codebook_dim", 512)
            num_semantic_quantizers = decoder_config.get("num_semantic_quantizers", 1)
            shared = {
                "codebook_size": decoder_config.get("codebook_size", 2048),
                "codebook_dim": input_width // 2,
                "vector_quantization_hidden_dimension": input_width // 2,
                "hidden_size": input_width,
            }
            if config_dict.get("acoustic_quantizer_config") is None:
                num_acoustic_quantizers = decoder_config.get("num_quantizers", 16) - num_semantic_quantizers
                config_dict["acoustic_quantizer_config"] = {**shared, "n_codebooks": num_acoustic_quantizers}
            if config_dict.get("semantic_quantizer_config") is None:
                config_dict["semantic_quantizer_config"] = {**shared, "n_codebooks": num_semantic_quantizers}
        for key in quantizer_keys:
            if key in kwargs:
                quantizer_config = deepcopy(kwargs.pop(key))
                if isinstance(config_dict.get(key), dict) and isinstance(quantizer_config, dict):
                    quantizer_config = {**config_dict[key], **quantizer_config}
                config_dict[key] = quantizer_config
        return super().from_dict(config_dict, **kwargs)

    @classmethod
    def from_json_file(cls, json_file):
        return cls.from_dict(cls._dict_from_json_file(json_file))

    @property
    def encoder_config(self):
        encoder_config = {
            name.removeprefix("encoder_"): deepcopy(getattr(self, name))
            for name in self.__dataclass_fields__
            if name.startswith("encoder_")
        }
        return Qwen3TTSTokenizerEncoderConfig(
            **encoder_config,
            sampling_rate=self.input_sampling_rate,
            dtype=self.dtype,
            attn_implementation=self._attn_implementation,
        )

    @property
    def decoder_config(self):
        decoder_config = {
            name: deepcopy(getattr(self, name)) for name in Qwen3TTSTokenizerDecoderConfig.__annotations__
        }
        return Qwen3TTSTokenizerDecoderConfig(
            **decoder_config, dtype=self.dtype, attn_implementation=self._attn_implementation
        )


class Qwen3TTSTokenizerCausalConvNet(Qwen3OmniMoeCausalConvNet):
    pass


class Qwen3TTSTokenizerCausalTransConvNet(Qwen3OmniMoeCausalTransConvNet):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__(in_channels, out_channels, kernel_size, stride)
        pad = kernel_size - stride
        self.left_pad = 0
        self.right_pad = int(pad)

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        if self.right_pad > 0:
            hidden_state = hidden_state[..., : hidden_state.shape[-1] - self.right_pad]
        return hidden_state.contiguous()


class Qwen3TTSTokenizerConvNeXtBlock(Qwen3OmniMoeConvNeXtBlock):
    pass


@auto_docstring
@dataclass
class Qwen3TTSTokenizerEncoderOutput(ModelOutput):
    r"""
    audio_codes (`torch.LongTensor` of shape `(batch_size, codes_length, num_quantizers)`):
        Discrete code indices computed using `model.encode`.
    audio_codes_mask (`torch.Tensor` of shape `(batch_size, codes_length)`, *optional*):
        Downsampled `padding_mask` indicating the valid frames in `audio_codes`.
    """

    audio_codes: torch.LongTensor | None = None
    audio_codes_mask: torch.Tensor | None = None


class Qwen3TTSTokenizerDecoderOutput(DacDecoderOutput):
    pass


@auto_docstring
@dataclass
class Qwen3TTSTokenizerOutput(ModelOutput):
    r"""
    audio_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
        Decoded audio waveform, truncated to the input length.
    audio_codes (`torch.LongTensor` of shape `(batch_size, codes_length, num_quantizers)`):
        Discrete code indices computed using `model.encode`.
    audio_codes_mask (`torch.Tensor` of shape `(batch_size, codes_length)`, *optional*):
        Downsampled `padding_mask` indicating the valid frames in `audio_codes`.
    """

    audio_values: torch.FloatTensor | None = None
    audio_codes: torch.LongTensor | None = None
    audio_codes_mask: torch.Tensor | None = None


class Qwen3TTSTokenizerPreTrainedModel(MimiPreTrainedModel):
    base_model_prefix = "model"
    _can_compile_fullgraph = False


@auto_docstring
class Qwen3TTSTokenizerCode2WavPreTrainedModel(Qwen3TTSTokenizerPreTrainedModel):
    config_class = Qwen3TTSTokenizerDecoderConfig
    _no_split_modules = ["Qwen3TTSTokenizerCode2WavTransformerLayer", "Qwen3TTSTokenizerDecoderBlock"]


class Qwen3TTSTokenizerDecoderBlock(Qwen3OmniMoeCode2WavDecoderBlock):
    def __init__(self, config: Qwen3TTSTokenizerDecoderConfig, layer_idx):
        super().__init__(config, layer_idx)


class Qwen3TTSTokenizerSnakeBeta(Qwen2_5OmniSnakeBeta):
    pass


class Qwen3TTSTokenizerEuclideanCodebook(MimiEuclideanCodebook):
    def __init__(self, config, epsilon: float = 1e-5):
        super().__init__(config, epsilon)
        # Not stored in the original checkpoint; `_init_weights` sets it.
        self.initialized = nn.Buffer(torch.tensor([True], dtype=torch.float32), persistent=False)


class Qwen3TTSTokenizerVectorQuantization(MimiVectorQuantization):
    pass


class Qwen3TTSTokenizerResidualVectorQuantizer(MimiResidualVectorQuantizer):
    def __init__(self, config, num_quantizers: int | None = None):
        super().__init__(config, num_quantizers)
        del self.frame_rate


class Qwen3TTSTokenizerSplitResidualVectorQuantizer(MimiSplitResidualVectorQuantizer):
    def __init__(self, acoustic_config, semantic_config):
        nn.Module.__init__(self)
        self.codebook_size = acoustic_config.codebook_size
        self.num_semantic_quantizers = semantic_config.n_codebooks
        self.num_acoustic_quantizers = acoustic_config.n_codebooks
        self.max_num_quantizers = self.num_semantic_quantizers + self.num_acoustic_quantizers
        self.semantic_residual_vector_quantizer = Qwen3TTSTokenizerResidualVectorQuantizer(
            semantic_config, self.num_semantic_quantizers
        )
        self.acoustic_residual_vector_quantizer = Qwen3TTSTokenizerResidualVectorQuantizer(
            acoustic_config, self.num_acoustic_quantizers
        )


class Qwen3TTSTokenizerCode2WavAttention(Qwen3OmniMoeCode2WavAttention):
    def __init__(self, config: Qwen3TTSTokenizerDecoderConfig, layer_idx):
        super().__init__(config, layer_idx)


class Qwen3TTSTokenizerCode2WavTransformerLayer(Qwen3OmniMoeCode2WavTransformerLayer):
    def __init__(self, config: Qwen3TTSTokenizerDecoderConfig, layer_idx):
        super().__init__(config, layer_idx)


class Qwen3TTSTokenizerDecoderTransformerModel(Qwen3OmniMoeCode2WavTransformerModel):
    def __init__(self, config: Qwen3TTSTokenizerDecoderConfig):
        super().__init__(config)


@auto_docstring
class Qwen3TTSTokenizerDecoderModel(Qwen3TTSTokenizerCode2WavPreTrainedModel):
    config_class = Qwen3TTSTokenizerDecoderConfig
    base_model_prefix = "decoder"

    def __init__(self, config: Qwen3TTSTokenizerDecoderConfig):
        super().__init__(config)
        self.total_upsample = int(np.prod(list(config.upsample_rates) + list(config.upsampling_ratios)))
        self.pre_transformer = Qwen3TTSTokenizerDecoderTransformerModel(config)
        self.input_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.latent_dim)

        self.pre_conv = Qwen3TTSTokenizerCausalConvNet(config.codebook_dim, config.latent_dim, kernel_size=3)

        upsample = []
        for factor in config.upsampling_ratios:
            upsample.append(
                nn.ModuleList(
                    [
                        Qwen3TTSTokenizerCausalTransConvNet(config.latent_dim, config.latent_dim, factor, factor),
                        Qwen3TTSTokenizerConvNeXtBlock(config.latent_dim),
                    ]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        decoder = [Qwen3TTSTokenizerCausalConvNet(config.latent_dim, config.decoder_dim, 7)]
        for i in range(len(config.upsample_rates)):
            decoder.append(Qwen3TTSTokenizerDecoderBlock(config, i))
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        decoder += [
            Qwen3TTSTokenizerSnakeBeta(output_dim),
            Qwen3TTSTokenizerCausalConvNet(output_dim, 1, 7),
        ]
        self.decoder = nn.ModuleList(decoder)
        self.post_init()

    @auto_docstring
    def forward(self, quantized_representation, **kwargs: Unpack[TransformersKwargs]):
        r"""
        quantized_representation (`torch.FloatTensor` of shape `(batch_size, codebook_dim, sequence_length)`):
            Quantized continuous representation to decode into waveform values.
        """
        chunk_size = self.config.decoder_chunk_size
        left_context_size = self.config.decoder_left_context_size
        wavs = []
        start_index = 0
        while start_index < quantized_representation.shape[-1]:
            end_index = min(start_index + chunk_size, quantized_representation.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            wav_chunk = self._decode_chunk(quantized_representation[..., start_index - context_size : end_index])
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)

    def _decode_chunk(self, quantized_representation):
        hidden = self.pre_conv(quantized_representation).transpose(1, 2)
        hidden = self.input_proj(hidden)
        hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
        hidden = self.output_proj(hidden)
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)


class Qwen3TTSTokenizerEncoder(MimiEncoder):
    pass


class Qwen3TTSTokenizerTransformerModel(MimiTransformerModel):
    pass


class Qwen3TTSTokenizerConv1d(MimiConv1d):
    pass


@auto_docstring(
    custom_intro="""
    The Qwen3TTSTokenizer encoder: Mimi's convolutional encoder, encoder transformer, and downsampling layer.
    """
)
class Qwen3TTSTokenizerEncoderModel(Qwen3TTSTokenizerPreTrainedModel):
    config_class = Qwen3TTSTokenizerEncoderConfig
    base_model_prefix = "encoder"

    def __init__(self, config):
        super().__init__(config)
        self.encoder = Qwen3TTSTokenizerEncoder(config)
        self.encoder_transformer = Qwen3TTSTokenizerTransformerModel(config)
        self.downsample = Qwen3TTSTokenizerConv1d(
            config,
            config.hidden_size,
            config.hidden_size,
            kernel_size=2 * int(config.encodec_frame_rate / config.frame_rate),
            stride=2,
            bias=False,
            pad_mode="replicate",
        )
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(self, input_values: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> BaseModelOutput:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input audio waveform.
        """
        embeddings = self.encoder(input_values)
        embeddings = self.encoder_transformer(embeddings.transpose(1, 2), use_cache=False)[0].transpose(1, 2)
        embeddings = self.downsample(embeddings)
        return BaseModelOutput(last_hidden_state=embeddings)


@auto_docstring
class Qwen3TTSTokenizerModel(VibeVoiceAcousticTokenizerModel, PreTrainedAudioTokenizerBase):
    config_class = Qwen3TTSTokenizerConfig
    main_input_name = "input_values"
    # The original checkpoint also stores the encoder's own copy of the quantizer, identical to `quantizer`.
    _keys_to_ignore_on_load_unexpected = [r"^encoder\.quantizer\."]

    def __init__(self, config: Qwen3TTSTokenizerConfig):
        super().__init__(config)
        self.quantizer = Qwen3TTSTokenizerSplitResidualVectorQuantizer(
            self.config.acoustic_quantizer_config, self.config.semantic_quantizer_config
        )

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ):
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Input audio waveform.
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Padding mask used to pad `input_values`.
        """
        embeddings = self.encoder(input_values).last_hidden_state
        audio_codes = self.quantizer.encode(embeddings).permute(1, 2, 0)

        audio_codes_mask = None
        if padding_mask is not None:
            frame_size = self.encoder.config.frame_size
            codes_length = (padding_mask.sum(dim=-1, keepdim=True) + frame_size - 1) // frame_size
            frame_index = torch.arange(audio_codes.shape[1], device=padding_mask.device).view(1, -1)
            audio_codes_mask = (frame_index < codes_length).to(padding_mask.dtype)

        return Qwen3TTSTokenizerEncoderOutput(audio_codes=audio_codes, audio_codes_mask=audio_codes_mask)

    @can_return_tuple
    @auto_docstring
    def decode(
        self,
        audio_codes: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3TTSTokenizerDecoderOutput:
        r"""
        audio_codes (`torch.LongTensor` of shape `(batch_size, codes_length, num_quantizers)`, *optional*):
            Discrete code indices computed using `model.encode`. Takes precedence over `latents` if both are provided.
        latents (`torch.Tensor` of shape `(batch_size, codebook_dim, codes_length)`, *optional*):
            Quantized continuous representation, using the decoder's input width. Bypasses the quantizer when
            `audio_codes` is not provided.
        """
        if latents is None and audio_codes is None:
            raise ValueError("Either `latents` or `audio_codes` must be provided.")

        if audio_codes is not None:
            latents = self.quantizer.decode(audio_codes.transpose(1, 2))
        audio_values = self.decoder(latents, **kwargs).squeeze(1)

        return Qwen3TTSTokenizerDecoderOutput(audio_values=audio_values)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3TTSTokenizerOutput:
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`):
            Input audio waveform.
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Padding mask used to pad `input_values`.
        """
        length = input_values.shape[-1]
        encoder_outputs = self.encode(input_values, padding_mask=padding_mask, return_dict=True)
        audio_values = self.decode(encoder_outputs.audio_codes, return_dict=True, **kwargs).audio_values[..., :length]

        return Qwen3TTSTokenizerOutput(
            audio_values=audio_values,
            audio_codes=encoder_outputs.audio_codes,
            audio_codes_mask=encoder_outputs.audio_codes_mask,
        )


__all__ = [
    "Qwen3TTSTokenizerConfig",
    "Qwen3TTSTokenizerEncoderConfig",
    "Qwen3TTSTokenizerDecoderConfig",
    "Qwen3TTSTokenizerAcousticQuantizerConfig",
    "Qwen3TTSTokenizerSemanticQuantizerConfig",
    "Qwen3TTSTokenizerModel",
    "Qwen3TTSTokenizerEncoderModel",
    "Qwen3TTSTokenizerDecoderModel",
    "Qwen3TTSTokenizerPreTrainedModel",
]
