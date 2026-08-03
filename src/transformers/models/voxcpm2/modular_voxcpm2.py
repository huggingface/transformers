# Copyright 2026 The OpenBMB Team and the HuggingFace Inc. team. All rights reserved.
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

import copy
import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn
from torch.func import jvp

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_bidirectional_mask, create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ..dac.modeling_dac import Snake1d
from ..minicpm4.configuration_minicpm4 import MiniCPM4Config
from ..minicpm4.modeling_minicpm4 import (
    MiniCPM4Attention,
    MiniCPM4DecoderLayer,
    MiniCPM4RMSNorm,
    MiniCPM4RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="openbmb/VoxCPM2")
@strict
class VoxCPM2TextConfig(MiniCPM4Config):
    r"""
    scale_emb (`int` or `float`, *optional*, defaults to 12):
        Multiplier applied to input embeddings.
    scale_depth (`int` or `float`, *optional*, defaults to 1.4):
        Multiplier for residual connections.
    dim_model_base (`int`, *optional*, defaults to 256):
        Base model dimension used to scale hidden states before the language model head.
    mup_denominator (`int`, *optional*):
        Width denominator used by compatible speculative decoding heads.
    sparse_config (`dict`, *optional*):
        Configuration for OpenBMB's optional InfLLM-v2 sparse attention implementation.
    use_mup (`bool`, *optional*, defaults to `False`):
        Whether to apply the muP embedding and residual scaling used by standalone MiniCPM4 checkpoints.
    kv_channels (`int`, *optional*, defaults to 128):
        Dimension of each attention key and value head in the original VoxCPM2 configuration.
    no_rope (`bool`, *optional*, defaults to `False`):
        Whether to disable rotary position embeddings. The residual language model enables this setting.

    Example:

    ```python
    >>> from transformers import VoxCPM2TextConfig

    >>> configuration = VoxCPM2TextConfig()
    >>> configuration.hidden_size
    2048
    ```
    """

    model_type = "voxcpm2_text"
    base_config_key = "lm_config"

    vocab_size: int = 73448
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int | None = 2
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-5
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    scale_emb: int | float = 12
    scale_depth: int | float | None = 1.4
    dim_model_base: int | None = 256
    use_mup: bool = False
    kv_channels: int | None = 128
    no_rope: bool = False

    def __post_init__(self, **kwargs):
        if self.head_dim is None:
            self.head_dim = self.kv_channels
        elif self.kv_channels is None:
            self.kv_channels = self.head_dim
        elif self.head_dim != self.kv_channels:
            raise ValueError(
                f"`head_dim` ({self.head_dim}) must match `kv_channels` ({self.kv_channels}) for VoxCPM2."
            )
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="openbmb/VoxCPM2")
@strict
class VoxCPM2EncoderConfig(PreTrainedConfig):
    r"""
    hidden_dim (`int`, *optional*, defaults to 1024):
        Dimension of the local audio encoder hidden states.
    ffn_dim (`int`, *optional*, defaults to 4096):
        Dimension of the local audio encoder feed-forward layers.
    num_heads (`int`, *optional*, defaults to 16):
        Number of attention heads in each local audio encoder layer.
    num_layers (`int`, *optional*, defaults to 12):
        Number of local audio encoder layers.
    kv_channels (`int`, *optional*, defaults to 128):
        Dimension of each attention key and value head.
    """

    model_type = "voxcpm2_encoder"
    base_config_key = "encoder_config"
    attribute_map = {
        "hidden_size": "hidden_dim",
        "intermediate_size": "ffn_dim",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "head_dim": "kv_channels",
    }

    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 12
    kv_channels: int | None = 128

    def __post_init__(self, **kwargs):
        if self.kv_channels is None:
            self.kv_channels = self.hidden_dim // self.num_heads
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        super().validate_architecture()
        if self.hidden_dim <= 0 or self.ffn_dim <= 0:
            raise ValueError("`hidden_dim` and `ffn_dim` must be strictly positive.")
        if self.num_heads <= 0 or self.num_layers <= 0:
            raise ValueError("`num_heads` and `num_layers` must be strictly positive.")
        if self.kv_channels is None or self.kv_channels <= 0:
            raise ValueError("`kv_channels` must be strictly positive.")


@auto_docstring(checkpoint="openbmb/VoxCPM2")
@strict
class VoxCPM2CfmConfig(PreTrainedConfig):
    r"""
    sigma_min (`float`, *optional*, defaults to 1e-6):
        Minimum noise scale used by conditional flow matching.
    solver (`str`, *optional*, defaults to `"euler"`):
        Numerical solver used for flow-matching inference.
    t_scheduler (`str`, *optional*, defaults to `"log-norm"`):
        Distribution used to sample flow-matching timesteps during training.
    training_cfg_rate (`float`, *optional*, defaults to 0.1):
        Probability of dropping the conditioning signal during training.
    inference_cfg_rate (`float`, *optional*, defaults to 2.0):
        Default classifier-free guidance scale stored by the released checkpoint.
    reg_loss_type (`str`, *optional*, defaults to `"l1"`):
        Regression loss used for the flow-matching objective.
    ratio_r_neq_t_range (`tuple[float, float]`, *optional*, defaults to `(0.25, 0.75)`):
        Sampling range for the probability that the conditioning and target timesteps differ.
    noise_cond_prob_range (`tuple[float, float]`, *optional*, defaults to `(0.0, 0.0)`):
        Sampling range for noisy conditioning probability.
    noise_cond_scale (`float`, *optional*, defaults to 0.0):
        Scale of the noise added to conditioning latents.
    """

    model_type = "voxcpm2_cfm"
    base_config_key = "cfm_config"

    sigma_min: int | float = 1e-6
    solver: str = "euler"
    t_scheduler: str = "log-norm"
    training_cfg_rate: int | float = 0.1
    inference_cfg_rate: int | float = 2.0
    reg_loss_type: str = "l1"
    ratio_r_neq_t_range: list[int | float] | tuple[int | float, ...] = (0.25, 0.75)
    noise_cond_prob_range: list[int | float] | tuple[int | float, ...] = (0.0, 0.0)
    noise_cond_scale: int | float = 0.0

    def validate_architecture(self):
        super().validate_architecture()
        if self.sigma_min < 0:
            raise ValueError("`sigma_min` must be non-negative.")
        for name in ("ratio_r_neq_t_range", "noise_cond_prob_range"):
            value = getattr(self, name)
            if len(value) != 2 or value[0] > value[1]:
                raise ValueError(f"`{name}` must contain two ordered values.")
            if name == "noise_cond_prob_range" and not 0 <= value[0] <= value[1] <= 1:
                raise ValueError("`noise_cond_prob_range` values must be between 0 and 1.")


@auto_docstring(checkpoint="openbmb/VoxCPM2")
@strict
class VoxCPM2DiTConfig(PreTrainedConfig):
    r"""
    hidden_dim (`int`, *optional*, defaults to 1024):
        Dimension of the local diffusion Transformer hidden states.
    ffn_dim (`int`, *optional*, defaults to 4096):
        Dimension of the local diffusion Transformer feed-forward layers.
    num_heads (`int`, *optional*, defaults to 16):
        Number of attention heads in each local diffusion Transformer layer.
    num_layers (`int`, *optional*, defaults to 12):
        Number of local diffusion Transformer layers.
    kv_channels (`int`, *optional*, defaults to 128):
        Dimension of each attention key and value head.
    mean_mode (`bool`, *optional*, defaults to `False`):
        Whether the flow estimator uses the conditional mean formulation.
    cfm_config (`VoxCPM2CfmConfig` or `dict`, *optional*):
        Conditional flow-matching configuration.
    """

    model_type = "voxcpm2_dit"
    base_config_key = "dit_config"
    sub_configs = {"cfm_config": VoxCPM2CfmConfig}
    attribute_map = {
        "hidden_size": "hidden_dim",
        "intermediate_size": "ffn_dim",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "head_dim": "kv_channels",
        "dit_mean_mode": "mean_mode",
    }

    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 12
    kv_channels: int | None = 128
    mean_mode: bool = False
    cfm_config: dict | VoxCPM2CfmConfig | None = None

    def __post_init__(self, **kwargs):
        if self.kv_channels is None:
            self.kv_channels = self.hidden_dim // self.num_heads
        if self.cfm_config is None:
            self.cfm_config = VoxCPM2CfmConfig()
        elif isinstance(self.cfm_config, dict):
            self.cfm_config = VoxCPM2CfmConfig(**self.cfm_config)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        super().validate_architecture()
        if self.hidden_dim <= 0 or self.ffn_dim <= 0:
            raise ValueError("`hidden_dim` and `ffn_dim` must be strictly positive.")
        if self.num_heads <= 0 or self.num_layers <= 0:
            raise ValueError("`num_heads` and `num_layers` must be strictly positive.")
        if self.kv_channels is None or self.kv_channels <= 0:
            raise ValueError("`kv_channels` must be strictly positive.")


@auto_docstring(checkpoint="openbmb/VoxCPM2")
@strict
class VoxCPM2AudioVAEConfig(PreTrainedConfig):
    r"""
    encoder_dim (`int`, *optional*, defaults to 128):
        Initial channel dimension of the audio encoder.
    encoder_rates (`tuple[int, ...]`, *optional*, defaults to `(2, 5, 8, 8)`):
        Downsampling strides used by the audio encoder.
    latent_dim (`int`, *optional*, defaults to 64):
        Dimension of the continuous audio latents.
    decoder_dim (`int`, *optional*, defaults to 2048):
        Initial channel dimension of the audio decoder.
    decoder_rates (`tuple[int, ...]`, *optional*, defaults to `(8, 6, 5, 2, 2, 2)`):
        Upsampling strides used by the audio decoder.
    depthwise (`bool`, *optional*, defaults to `True`):
        Whether residual convolutions use depthwise groups.
    sample_rate (`int`, *optional*, defaults to 16000):
        Sampling rate expected by the audio encoder.
    out_sample_rate (`int`, *optional*, defaults to 48000):
        Sampling rate produced by the audio decoder.
    use_noise_block (`bool`, *optional*, defaults to `False`):
        Whether decoder blocks inject learned noise.
    sr_bin_boundaries (`tuple[int, ...]`, *optional*, defaults to `(20000, 30000, 40000)`):
        Boundaries used to bucket the requested decoder sampling rate.
    cond_type (`str`, *optional*, defaults to `"scale_bias"`):
        Type of sample-rate conditioning applied by the decoder.
    cond_dim (`int`, *optional*, defaults to 128):
        Embedding dimension used for sample-rate conditioning.
    cond_out_layer (`bool`, *optional*, defaults to `False`):
        Whether to condition the decoder output layer.
    """

    model_type = "voxcpm2_audio_vae"
    base_config_key = "audio_vae_config"

    encoder_dim: int = 128
    encoder_rates: list[int] | tuple[int, ...] = (2, 5, 8, 8)
    latent_dim: int = 64
    decoder_dim: int = 2048
    decoder_rates: list[int] | tuple[int, ...] = (8, 6, 5, 2, 2, 2)
    depthwise: bool = True
    sample_rate: int = 16000
    out_sample_rate: int = 48000
    use_noise_block: bool = False
    sr_bin_boundaries: list[int] | tuple[int, ...] | None = (20000, 30000, 40000)
    cond_type: str = "scale_bias"
    cond_dim: int = 128
    cond_out_layer: bool = False

    @property
    def hop_length(self) -> int:
        return math.prod(self.encoder_rates)

    @property
    def decode_hop_length(self) -> int:
        return math.prod(self.decoder_rates)

    def validate_architecture(self):
        super().validate_architecture()
        if self.encoder_dim <= 0 or self.decoder_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("AudioVAE dimensions must be strictly positive.")
        if self.sample_rate <= 0 or self.out_sample_rate <= 0:
            raise ValueError("AudioVAE sampling rates must be strictly positive.")
        if not self.encoder_rates or not self.decoder_rates:
            raise ValueError("AudioVAE encoder and decoder rates cannot be empty.")
        if any(rate <= 0 for rate in (*self.encoder_rates, *self.decoder_rates)):
            raise ValueError("AudioVAE encoder and decoder rates must be strictly positive.")


@auto_docstring(checkpoint="openbmb/VoxCPM2")
@strict
class VoxCPM2Config(PreTrainedConfig):
    r"""
    lm_config (`VoxCPM2TextConfig` or `dict`, *optional*):
        Configuration of the MiniCPM4-based text-semantic language model.
    encoder_config (`VoxCPM2EncoderConfig` or `dict`, *optional*):
        Configuration of the local audio encoder.
    dit_config (`VoxCPM2DiTConfig` or `dict`, *optional*):
        Configuration of the local diffusion Transformer and conditional flow matcher.
    audio_vae_config (`VoxCPM2AudioVAEConfig` or `dict`, *optional*):
        Configuration of AudioVAE V2.
    patch_size (`int`, *optional*, defaults to 4):
        Number of AudioVAE latent frames generated in each autoregressive step.
    feat_dim (`int`, *optional*, defaults to 64):
        Dimension of each AudioVAE latent frame.
    residual_lm_num_layers (`int`, *optional*, defaults to 8):
        Number of layers in the residual autoregressive language model.
    residual_lm_no_rope (`bool`, *optional*, defaults to `True`):
        Whether to disable rotary position embeddings in the residual language model.
    scalar_quantization_latent_dim (`int`, *optional*, defaults to 512):
        Hidden dimension of the scalar-quantization projection.
    scalar_quantization_scale (`int`, *optional*, defaults to 9):
        Number of scalar quantization levels on each side of zero.
    max_cache_length (`int`, *optional*, defaults to 8192):
        Maximum cache capacity used by the original streaming generation implementation.
    audio_start_token_id (`int`, *optional*, defaults to 101):
        Token marking the beginning of generated audio context.
    audio_end_token_id (`int`, *optional*, defaults to 102):
        Token marking the end of generated audio context.
    reference_audio_start_token_id (`int`, *optional*, defaults to 103):
        Token marking the beginning of reference audio context.
    reference_audio_end_token_id (`int`, *optional*, defaults to 104):
        Token marking the end of reference audio context.

    Example:

    ```python
    >>> from transformers import VoxCPM2Config

    >>> configuration = VoxCPM2Config()
    >>> configuration.audio_vae_config.out_sample_rate
    48000
    ```
    """

    model_type = "voxcpm2"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {
        "lm_config": VoxCPM2TextConfig,
        "encoder_config": VoxCPM2EncoderConfig,
        "dit_config": VoxCPM2DiTConfig,
        "audio_vae_config": VoxCPM2AudioVAEConfig,
    }

    lm_config: dict | VoxCPM2TextConfig | None = None
    encoder_config: dict | VoxCPM2EncoderConfig | None = None
    dit_config: dict | VoxCPM2DiTConfig | None = None
    audio_vae_config: dict | VoxCPM2AudioVAEConfig | None = None
    patch_size: int = 4
    feat_dim: int = 64
    residual_lm_num_layers: int = 8
    residual_lm_no_rope: bool = True
    scalar_quantization_latent_dim: int = 512
    scalar_quantization_scale: int = 9
    max_cache_length: int = 8192
    audio_start_token_id: int = 101
    audio_end_token_id: int = 102
    reference_audio_start_token_id: int = 103
    reference_audio_end_token_id: int = 104

    def __post_init__(self, **kwargs):
        if self.lm_config is None:
            self.lm_config = VoxCPM2TextConfig()
        elif isinstance(self.lm_config, dict):
            self.lm_config = VoxCPM2TextConfig(**self.lm_config)

        if self.encoder_config is None:
            self.encoder_config = VoxCPM2EncoderConfig()
        elif isinstance(self.encoder_config, dict):
            self.encoder_config = VoxCPM2EncoderConfig(**self.encoder_config)

        if self.dit_config is None:
            self.dit_config = VoxCPM2DiTConfig()
        elif isinstance(self.dit_config, dict):
            self.dit_config = VoxCPM2DiTConfig(**self.dit_config)

        if self.audio_vae_config is None:
            self.audio_vae_config = VoxCPM2AudioVAEConfig()
        elif isinstance(self.audio_vae_config, dict):
            self.audio_vae_config = VoxCPM2AudioVAEConfig(**self.audio_vae_config)

        if (legacy_max_length := kwargs.pop("max_length", None)) is not None:
            self.max_cache_length = legacy_max_length
        kwargs.pop("architecture", None)
        kwargs.pop("device", None)

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        super().validate_architecture()
        if self.patch_size <= 0 or self.residual_lm_num_layers <= 0 or self.max_cache_length <= 0:
            raise ValueError("Patch size, residual LM layers, and cache length must be strictly positive.")
        if self.feat_dim != self.audio_vae_config.latent_dim:
            raise ValueError(
                f"`feat_dim` ({self.feat_dim}) must match AudioVAE `latent_dim` ({self.audio_vae_config.latent_dim})."
            )

    def get_text_config(self, *args, **kwargs):
        return self.lm_config


def _get_tslm_config(config: VoxCPM2Config) -> VoxCPM2TextConfig:
    if not isinstance(config.lm_config, VoxCPM2TextConfig):
        raise TypeError("`lm_config` must be a `VoxCPM2TextConfig` instance")
    return copy.deepcopy(config.lm_config)


def _get_ralm_config(config: VoxCPM2Config) -> VoxCPM2TextConfig:
    ralm_config = _get_tslm_config(config)
    ralm_config.num_hidden_layers = config.residual_lm_num_layers
    ralm_config.vocab_size = 0
    ralm_config.no_rope = config.residual_lm_no_rope
    return ralm_config


def _get_local_encoder_backbone_config(config: VoxCPM2Config) -> VoxCPM2TextConfig:
    if not isinstance(config.encoder_config, VoxCPM2EncoderConfig):
        raise TypeError("`encoder_config` must be a `VoxCPM2EncoderConfig` instance")
    encoder_config = _get_tslm_config(config)
    encoder_config.hidden_size = config.encoder_config.hidden_dim
    encoder_config.intermediate_size = config.encoder_config.ffn_dim
    encoder_config.num_attention_heads = config.encoder_config.num_heads
    encoder_config.num_hidden_layers = config.encoder_config.num_layers
    encoder_config.kv_channels = config.encoder_config.kv_channels
    encoder_config.head_dim = config.encoder_config.kv_channels
    encoder_config.vocab_size = 0
    return encoder_config


def _get_local_dit_backbone_config(config: VoxCPM2Config) -> VoxCPM2TextConfig:
    if not isinstance(config.dit_config, VoxCPM2DiTConfig):
        raise TypeError("`dit_config` must be a `VoxCPM2DiTConfig` instance")
    dit_config = _get_tslm_config(config)
    dit_config.hidden_size = config.dit_config.hidden_dim
    dit_config.intermediate_size = config.dit_config.ffn_dim
    dit_config.num_attention_heads = config.dit_config.num_heads
    dit_config.num_hidden_layers = config.dit_config.num_layers
    dit_config.kv_channels = config.dit_config.kv_channels
    dit_config.head_dim = config.dit_config.kv_channels
    dit_config.vocab_size = 0
    return dit_config


class VoxCPM2ScalarQuantizationLayer(nn.Module):
    def __init__(self, config: VoxCPM2Config):
        super().__init__()
        self.in_dim = config.lm_config.hidden_size
        self.out_dim = config.lm_config.hidden_size
        self.latent_dim = config.scalar_quantization_latent_dim
        self.scale = config.scalar_quantization_scale

        self.in_proj = nn.Linear(self.in_dim, self.latent_dim)
        self.out_proj = nn.Linear(self.latent_dim, self.out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.tanh(self.in_proj(hidden_states))
        quantized_states = torch.round(hidden_states * self.scale) / self.scale
        if self.training:
            hidden_states = hidden_states + (quantized_states - hidden_states).detach()
        else:
            hidden_states = quantized_states
        return self.out_proj(hidden_states)


class VoxCPM2Snake1d(Snake1d):
    pass


class VoxCPM2CausalConv1d(nn.Conv1d):
    def __init__(self, *args, padding: int = 0, output_padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = padding * 2 - output_padding

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, (self.causal_padding, 0))
        return super().forward(hidden_states)


class VoxCPM2CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, padding: int = 0, output_padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_trim = padding * 2 - output_padding

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = super().forward(hidden_states)
        if self.causal_trim > 0:
            hidden_states = hidden_states[..., : -self.causal_trim]
        return hidden_states


def _apply_voxcpm2_weight_norm(module: nn.Module) -> nn.Module:
    weight_norm = nn.utils.weight_norm
    if hasattr(nn.utils.parametrizations, "weight_norm"):
        weight_norm = nn.utils.parametrizations.weight_norm
    return weight_norm(module)


class VoxCPM2CausalResidualUnit(nn.Module):
    def __init__(self, hidden_dim: int = 16, dilation: int = 1, kernel_size: int = 7, groups: int = 1):
        super().__init__()
        padding = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            VoxCPM2Snake1d(hidden_dim),
            _apply_voxcpm2_weight_norm(
                VoxCPM2CausalConv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    groups=groups,
                )
            ),
            VoxCPM2Snake1d(hidden_dim),
            _apply_voxcpm2_weight_norm(VoxCPM2CausalConv1d(hidden_dim, hidden_dim, kernel_size=1)),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.block(hidden_states)


class VoxCPM2CausalEncoderBlock(nn.Module):
    def __init__(self, output_dim: int = 16, input_dim: int | None = None, stride: int = 1, groups: int = 1):
        super().__init__()
        input_dim = input_dim or output_dim // 2
        self.block = nn.Sequential(
            VoxCPM2CausalResidualUnit(input_dim, dilation=1, groups=groups),
            VoxCPM2CausalResidualUnit(input_dim, dilation=3, groups=groups),
            VoxCPM2CausalResidualUnit(input_dim, dilation=9, groups=groups),
            VoxCPM2Snake1d(input_dim),
            _apply_voxcpm2_weight_norm(
                VoxCPM2CausalConv1d(
                    input_dim,
                    output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                    output_padding=stride % 2,
                )
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class VoxCPM2AudioEncoder(nn.Module):
    def __init__(self, config: VoxCPM2AudioVAEConfig):
        super().__init__()
        hidden_dim = config.encoder_dim
        layers = [_apply_voxcpm2_weight_norm(VoxCPM2CausalConv1d(1, hidden_dim, kernel_size=7, padding=3))]
        for stride in config.encoder_rates:
            hidden_dim *= 2
            groups = hidden_dim // 2 if config.depthwise else 1
            layers.append(VoxCPM2CausalEncoderBlock(output_dim=hidden_dim, stride=stride, groups=groups))

        self.fc_mu = _apply_voxcpm2_weight_norm(
            VoxCPM2CausalConv1d(hidden_dim, config.latent_dim, kernel_size=3, padding=1)
        )
        self.fc_logvar = _apply_voxcpm2_weight_norm(
            VoxCPM2CausalConv1d(hidden_dim, config.latent_dim, kernel_size=3, padding=1)
        )
        self.block = nn.Sequential(*layers)
        self.encoder_dim = hidden_dim

    def forward(self, input_values: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden_states = self.block(input_values)
        return {
            "hidden_state": hidden_states,
            "mu": self.fc_mu(hidden_states),
            "logvar": self.fc_logvar(hidden_states),
        }


class VoxCPM2NoiseBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = _apply_voxcpm2_weight_norm(
            VoxCPM2CausalConv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(
            (hidden_states.shape[0], 1, hidden_states.shape[2]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return hidden_states + noise * self.linear(hidden_states)


class VoxCPM2CausalDecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        groups: int = 1,
        use_noise_block: bool = False,
    ):
        super().__init__()
        layers = [
            VoxCPM2Snake1d(input_dim),
            _apply_voxcpm2_weight_norm(
                VoxCPM2CausalConvTranspose1d(
                    input_dim,
                    output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                    output_padding=stride % 2,
                )
            ),
        ]
        if use_noise_block:
            layers.append(VoxCPM2NoiseBlock(output_dim))
        layers.extend(
            [
                VoxCPM2CausalResidualUnit(output_dim, dilation=1, groups=groups),
                VoxCPM2CausalResidualUnit(output_dim, dilation=3, groups=groups),
                VoxCPM2CausalResidualUnit(output_dim, dilation=9, groups=groups),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.input_channels = input_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class VoxCPM2SampleRateConditionLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_sample_rate_buckets: int,
        conditioning_type: str = "scale_bias",
        conditioning_dim: int = 128,
        use_output_layer: bool = False,
    ):
        super().__init__()
        self.conditioning_type = conditioning_type
        output_layer_input_dim = input_dim

        if conditioning_type in ("scale_bias", "scale_bias_init"):
            self.scale_embed = nn.Embedding(num_sample_rate_buckets, input_dim)
            self.bias_embed = nn.Embedding(num_sample_rate_buckets, input_dim)
            if conditioning_type == "scale_bias":
                nn.init.ones_(self.scale_embed.weight)
                nn.init.zeros_(self.bias_embed.weight)
            else:
                nn.init.normal_(self.scale_embed.weight, mean=1.0)
                nn.init.normal_(self.bias_embed.weight)
        elif conditioning_type == "add":
            self.cond_embed = nn.Embedding(num_sample_rate_buckets, input_dim)
            nn.init.normal_(self.cond_embed.weight)
        elif conditioning_type == "concat":
            if not use_output_layer:
                raise ValueError("`use_output_layer` must be enabled for concatenated sample-rate conditioning")
            self.cond_embed = nn.Embedding(num_sample_rate_buckets, conditioning_dim)
            output_layer_input_dim = input_dim + conditioning_dim
        else:
            raise ValueError(f"Invalid sample-rate conditioning type: {conditioning_type}")

        if use_output_layer:
            self.out_layer = nn.Sequential(
                VoxCPM2Snake1d(output_layer_input_dim),
                _apply_voxcpm2_weight_norm(VoxCPM2CausalConv1d(output_layer_input_dim, input_dim, kernel_size=1)),
            )
        else:
            self.out_layer = nn.Identity()

    def forward(self, hidden_states: torch.Tensor, sample_rate_ids: torch.LongTensor) -> torch.Tensor:
        if self.conditioning_type in ("scale_bias", "scale_bias_init"):
            hidden_states = hidden_states * self.scale_embed(sample_rate_ids).unsqueeze(-1)
            hidden_states = hidden_states + self.bias_embed(sample_rate_ids).unsqueeze(-1)
        elif self.conditioning_type == "add":
            hidden_states = hidden_states + self.cond_embed(sample_rate_ids).unsqueeze(-1)
        else:
            conditioning = self.cond_embed(sample_rate_ids).unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
            hidden_states = torch.cat((hidden_states, conditioning), dim=1)
        return self.out_layer(hidden_states)


class VoxCPM2AudioDecoder(nn.Module):
    def __init__(self, config: VoxCPM2AudioVAEConfig):
        super().__init__()
        if config.depthwise:
            layers = [
                _apply_voxcpm2_weight_norm(
                    VoxCPM2CausalConv1d(
                        config.latent_dim,
                        config.latent_dim,
                        kernel_size=7,
                        padding=3,
                        groups=config.latent_dim,
                    )
                ),
                _apply_voxcpm2_weight_norm(VoxCPM2CausalConv1d(config.latent_dim, config.decoder_dim, kernel_size=1)),
            ]
        else:
            layers = [
                _apply_voxcpm2_weight_norm(
                    VoxCPM2CausalConv1d(config.latent_dim, config.decoder_dim, kernel_size=7, padding=3)
                )
            ]

        for stride_index, stride in enumerate(config.decoder_rates):
            input_dim = config.decoder_dim // 2**stride_index
            output_dim = config.decoder_dim // 2 ** (stride_index + 1)
            groups = output_dim if config.depthwise else 1
            layers.append(
                VoxCPM2CausalDecoderBlock(
                    input_dim,
                    output_dim,
                    stride,
                    groups=groups,
                    use_noise_block=config.use_noise_block,
                )
            )

        layers.extend(
            [
                VoxCPM2Snake1d(output_dim),
                _apply_voxcpm2_weight_norm(VoxCPM2CausalConv1d(output_dim, 1, kernel_size=7, padding=3)),
                nn.Tanh(),
            ]
        )

        if config.sr_bin_boundaries is None:
            self.model = nn.Sequential(*layers)
            self.sr_bin_boundaries = None
        else:
            self.model = nn.ModuleList(layers)
            self.register_buffer(
                "sr_bin_boundaries", torch.tensor(config.sr_bin_boundaries, dtype=torch.int32), persistent=True
            )
            num_sample_rate_buckets = len(config.sr_bin_boundaries) + 1
            conditioning_layers = []
            for layer in self.model:
                if isinstance(layer, VoxCPM2CausalDecoderBlock):
                    conditioning_layers.append(
                        VoxCPM2SampleRateConditionLayer(
                            input_dim=layer.input_channels,
                            num_sample_rate_buckets=num_sample_rate_buckets,
                            conditioning_type=config.cond_type,
                            conditioning_dim=config.cond_dim,
                            use_output_layer=config.cond_out_layer,
                        )
                    )
                else:
                    conditioning_layers.append(None)
            self.sr_cond_model = nn.ModuleList(conditioning_layers)

    def get_sample_rate_ids(self, sample_rate: torch.Tensor) -> torch.Tensor:
        return torch.bucketize(sample_rate, self.sr_bin_boundaries)

    def forward(self, hidden_states: torch.Tensor, sample_rate: torch.Tensor | None = None) -> torch.Tensor:
        if self.sr_bin_boundaries is None:
            return self.model(hidden_states)
        if sample_rate is None:
            raise ValueError("`sample_rate` must be provided when sample-rate conditioning is enabled")

        sample_rate_ids = self.get_sample_rate_ids(sample_rate)
        for layer, conditioning_layer in zip(self.model, self.sr_cond_model):
            if conditioning_layer is not None:
                hidden_states = conditioning_layer(hidden_states, sample_rate_ids)
            hidden_states = layer(hidden_states)
        return hidden_states


class VoxCPM2AudioVAE(nn.Module):
    def __init__(self, config: VoxCPM2AudioVAEConfig):
        super().__init__()
        self.config = config
        self.encoder_dim = config.encoder_dim
        self.encoder_rates = config.encoder_rates
        self.decoder_dim = config.decoder_dim
        self.decoder_rates = config.decoder_rates
        self.depthwise = config.depthwise
        self.use_noise_block = config.use_noise_block
        self.latent_dim = config.latent_dim
        self.hop_length = config.hop_length
        self.encoder = VoxCPM2AudioEncoder(config)
        self.decoder = VoxCPM2AudioDecoder(config)
        self.sample_rate = config.sample_rate
        self.out_sample_rate = config.out_sample_rate
        self.sr_bin_boundaries = config.sr_bin_boundaries
        self.chunk_size = config.hop_length
        self.decode_chunk_size = config.decode_hop_length

    def preprocess(self, input_values: torch.Tensor, sampling_rate: int | None = None) -> torch.Tensor:
        sampling_rate = self.sample_rate if sampling_rate is None else sampling_rate
        if sampling_rate != self.sample_rate:
            raise ValueError(f"VoxCPM2 AudioVAE expects {self.sample_rate} Hz audio, but received {sampling_rate} Hz")
        right_padding = math.ceil(input_values.shape[-1] / self.hop_length) * self.hop_length
        right_padding -= input_values.shape[-1]
        return F.pad(input_values, (0, right_padding))

    def encode(self, input_values: torch.Tensor, sampling_rate: int | None = None) -> torch.Tensor:
        if input_values.ndim == 2:
            input_values = input_values.unsqueeze(1)
        input_values = self.preprocess(input_values, sampling_rate)
        return self.encoder(input_values)["mu"]


class VoxCPM2SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        if embedding_dim < 4 or embedding_dim % 2 != 0:
            raise ValueError("`embedding_dim` must be an even integer greater than 2.")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        half_dim = self.embedding_dim // 2
        exponent = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, dtype=timesteps.dtype, device=timesteps.device) * -exponent)
        embeddings = scale * timesteps.unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class VoxCPM2TimestepEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int | None = None):
        super().__init__()
        output_dim = output_dim if output_dim is not None else hidden_dim
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.linear_2(hidden_states)


class VoxCPM2Attention(MiniCPM4Attention):
    def __init__(self, config: VoxCPM2TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        is_causal: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_causal=is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class VoxCPM2RMSNorm(MiniCPM4RMSNorm):
    pass


class VoxCPM2RotaryEmbedding(MiniCPM4RotaryEmbedding):
    def __init__(self, config: VoxCPM2TextConfig, device=None):
        super().__init__(config, device)


class VoxCPM2DecoderLayer(MiniCPM4DecoderLayer):
    def __init__(self, config: VoxCPM2TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.residual_scale = config.scale_depth / math.sqrt(config.num_hidden_layers) if config.use_mup else 1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        is_causal: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            is_causal=is_causal,
            **kwargs,
        )
        hidden_states = residual + hidden_states * self.residual_scale

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states * self.residual_scale


class VoxCPM2BackboneModel(nn.Module):
    def __init__(self, config: VoxCPM2TextConfig):
        super().__init__()
        if config.sparse_config is not None:
            raise NotImplementedError(
                "VoxCPM2 InfLLM-v2 sparse attention is not implemented in Transformers. Remove `sparse_config` to "
                "use dense attention."
            )
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = (
            nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            if config.vocab_size > 0
            else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [VoxCPM2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = VoxCPM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = None if config.no_rope else VoxCPM2RotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool = False,
        is_causal: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            if self.vocab_size == 0:
                raise ValueError("`inputs_embeds` must be provided when `vocab_size` is 0")
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            ).unsqueeze(0)

        if is_causal:
            attention_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        else:
            attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        hidden_states = inputs_embeds
        position_embeddings = (
            None if self.rotary_emb is None else self.rotary_emb(hidden_states, position_ids=position_ids)
        )
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                is_causal=is_causal,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class VoxCPM2LocalEncoder(nn.Module):
    def __init__(self, config: VoxCPM2Config):
        super().__init__()
        encoder_config = _get_local_encoder_backbone_config(config)
        self.special_token = nn.Parameter(torch.randn(1, 1, 1, encoder_config.hidden_size))
        self.in_proj = nn.Linear(config.feat_dim, encoder_config.hidden_size)
        self.encoder = VoxCPM2BackboneModel(encoder_config)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_steps, _, _ = audio_features.shape
        hidden_states = self.in_proj(audio_features)
        special_tokens = self.special_token.expand(batch_size, num_steps, 1, -1)
        hidden_states = torch.cat((special_tokens, hidden_states), dim=2)
        hidden_states = hidden_states.reshape(batch_size * num_steps, hidden_states.shape[2], hidden_states.shape[3])
        hidden_states = self.encoder(inputs_embeds=hidden_states, is_causal=False).last_hidden_state[:, 0]
        return hidden_states.reshape(batch_size, num_steps, -1)


class VoxCPM2LocalDiT(nn.Module):
    def __init__(self, config: VoxCPM2Config):
        super().__init__()
        dit_config = _get_local_dit_backbone_config(config)
        self.in_channels = config.feat_dim
        self.out_channels = config.feat_dim

        self.in_proj = nn.Linear(self.in_channels, dit_config.hidden_size)
        self.cond_proj = nn.Linear(self.in_channels, dit_config.hidden_size)
        self.out_proj = nn.Linear(dit_config.hidden_size, self.out_channels)
        self.time_embeddings = VoxCPM2SinusoidalPositionEmbedding(dit_config.hidden_size)
        self.time_mlp = VoxCPM2TimestepEmbedding(dit_config.hidden_size, dit_config.hidden_size)
        self.delta_time_mlp = VoxCPM2TimestepEmbedding(dit_config.hidden_size, dit_config.hidden_size)
        self.decoder = VoxCPM2BackboneModel(dit_config)

    def forward(
        self,
        sample: torch.Tensor,
        mu: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: torch.Tensor,
        delta_timestep: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.in_proj(sample.transpose(1, 2).contiguous())
        conditioning_hidden_states = self.cond_proj(conditioning.transpose(1, 2).contiguous())

        timestep_hidden_states = self.time_embeddings(timestep).to(hidden_states.dtype)
        timestep_hidden_states = self.time_mlp(timestep_hidden_states)
        delta_timestep_hidden_states = self.time_embeddings(delta_timestep).to(hidden_states.dtype)
        delta_timestep_hidden_states = self.delta_time_mlp(delta_timestep_hidden_states)
        timestep_hidden_states = timestep_hidden_states + delta_timestep_hidden_states

        mu_hidden_states = mu.reshape(hidden_states.shape[0], -1, hidden_states.shape[-1])
        hidden_states = torch.cat(
            (mu_hidden_states, timestep_hidden_states.unsqueeze(1), conditioning_hidden_states, hidden_states), dim=1
        )
        hidden_states = self.decoder(inputs_embeds=hidden_states, is_causal=False).last_hidden_state
        prefix_length = mu_hidden_states.shape[1] + 1 + conditioning_hidden_states.shape[1]
        hidden_states = self.out_proj(hidden_states[:, prefix_length:])
        return hidden_states.transpose(1, 2).contiguous()


class VoxCPM2ConditionalFlowMatching(nn.Module):
    def __init__(self, config: VoxCPM2Config):
        super().__init__()
        if not isinstance(config.dit_config, VoxCPM2DiTConfig) or not isinstance(
            config.dit_config.cfm_config, VoxCPM2CfmConfig
        ):
            raise TypeError("`dit_config.cfm_config` must be a `VoxCPM2CfmConfig` instance")
        cfm_config = config.dit_config.cfm_config
        self.solver = cfm_config.solver
        self.sigma_min = cfm_config.sigma_min
        self.t_scheduler = cfm_config.t_scheduler
        self.training_cfg_rate = cfm_config.training_cfg_rate
        self.inference_cfg_rate = cfm_config.inference_cfg_rate
        self.reg_loss_type = cfm_config.reg_loss_type
        self.ratio_r_neq_t_range = cfm_config.ratio_r_neq_t_range
        self.noise_cond_prob_range = cfm_config.noise_cond_prob_range
        self.noise_cond_scale = cfm_config.noise_cond_scale
        self.in_channels = config.feat_dim
        self.mean_mode = config.dit_config.mean_mode
        self.estimator = VoxCPM2LocalDiT(config)

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        num_inference_steps: int,
        patch_size: int,
        conditioning: torch.Tensor,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        sway_sampling_coefficient: float = 1.0,
        use_cfg_zero_star: bool = True,
    ) -> torch.Tensor:
        if self.solver != "euler":
            raise ValueError(f"Unsupported flow-matching solver: {self.solver}")
        sample = (
            torch.randn((mu.shape[0], self.in_channels, patch_size), device=mu.device, dtype=mu.dtype) * temperature
        )
        timestep_span = torch.linspace(1, 0, num_inference_steps + 1, device=mu.device, dtype=mu.dtype)
        timestep_span = timestep_span + sway_sampling_coefficient * (
            torch.cos(torch.pi / 2 * timestep_span) - 1 + timestep_span
        )
        return self.solve_euler(
            sample=sample,
            timestep_span=timestep_span,
            mu=mu,
            conditioning=conditioning,
            cfg_value=cfg_value,
            use_cfg_zero_star=use_cfg_zero_star,
        )

    def optimized_scale(self, positive_states: torch.Tensor, negative_states: torch.Tensor) -> torch.Tensor:
        dot_product = torch.sum(positive_states * negative_states, dim=1, keepdim=True)
        squared_norm = torch.sum(negative_states**2, dim=1, keepdim=True) + 1e-8
        return dot_product / squared_norm

    def solve_euler(
        self,
        sample: torch.Tensor,
        timestep_span: torch.Tensor,
        mu: torch.Tensor,
        conditioning: torch.Tensor,
        cfg_value: float = 1.0,
        use_cfg_zero_star: bool = True,
    ) -> torch.Tensor:
        timestep = timestep_span[0]
        delta_timestep = timestep_span[0] - timestep_span[1]
        zero_init_steps = max(1, int(len(timestep_span) * 0.04))

        for step in range(1, len(timestep_span)):
            if use_cfg_zero_star and step <= zero_init_steps:
                derivative = torch.zeros_like(sample)
            else:
                batch_size = sample.shape[0]
                sample_input = torch.cat((sample, sample), dim=0)
                mu_input = torch.cat((mu, torch.zeros_like(mu)), dim=0)
                timestep_input = timestep.expand(2 * batch_size)
                delta_timestep_input = delta_timestep.expand(2 * batch_size)
                if not self.mean_mode:
                    delta_timestep_input = torch.zeros_like(delta_timestep_input)
                conditioning_input = torch.cat((conditioning, conditioning), dim=0)

                positive_derivative, negative_derivative = self.estimator(
                    sample_input,
                    mu_input,
                    timestep_input,
                    conditioning_input,
                    delta_timestep_input,
                ).chunk(2)
                if use_cfg_zero_star:
                    optimized_scale = self.optimized_scale(
                        positive_derivative.reshape(batch_size, -1), negative_derivative.reshape(batch_size, -1)
                    )
                    optimized_scale = optimized_scale.reshape(batch_size, *([1] * (positive_derivative.ndim - 1)))
                else:
                    optimized_scale = 1.0
                derivative = negative_derivative * optimized_scale + cfg_value * (
                    positive_derivative - negative_derivative * optimized_scale
                )

            sample = sample - delta_timestep * derivative
            timestep = timestep - delta_timestep
            if step < len(timestep_span) - 1:
                delta_timestep = timestep - timestep_span[step + 1]

        return sample

    def adaptive_loss_weighting(
        self,
        losses: torch.Tensor,
        mask: torch.Tensor | None = None,
        power: float = 0.0,
        epsilon: float = 1e-3,
    ) -> torch.Tensor:
        weights = 1.0 / (losses + epsilon).pow(power)
        if mask is not None:
            weights = weights * mask
        return weights.detach()

    def sample_r_t(
        self,
        hidden_states: torch.Tensor,
        mean: float = -0.4,
        standard_deviation: float = 1.0,
        ratio_r_neq_t: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        if self.t_scheduler == "log-norm":
            r_samples = (
                torch.randn(batch_size, device=hidden_states.device, dtype=hidden_states.dtype) * standard_deviation
                + mean
            )
            t_samples = (
                torch.randn(batch_size, device=hidden_states.device, dtype=hidden_states.dtype) * standard_deviation
                + mean
            )
            r_samples = torch.sigmoid(r_samples)
            t_samples = torch.sigmoid(t_samples)
        elif self.t_scheduler == "uniform":
            r_samples = torch.rand(batch_size, device=hidden_states.device, dtype=hidden_states.dtype)
            t_samples = torch.rand(batch_size, device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            raise ValueError(f"Unsupported timestep scheduler: {self.t_scheduler}")

        use_distinct_timesteps = (
            torch.rand(batch_size, device=hidden_states.device, dtype=hidden_states.dtype) < ratio_r_neq_t
        )
        r_samples, t_samples = torch.where(
            use_distinct_timesteps,
            torch.stack((torch.minimum(r_samples, t_samples), torch.maximum(r_samples, t_samples))),
            torch.stack((t_samples, t_samples)),
        )
        return r_samples.squeeze(), t_samples.squeeze()

    def compute_loss(
        self,
        target: torch.Tensor,
        mu: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        progress: float = 0.0,
    ) -> torch.Tensor:
        batch_size = target.shape[0]
        if self.training_cfg_rate > 0:
            keep_conditioning = torch.rand(batch_size, device=target.device) > self.training_cfg_rate
            mu = mu * keep_conditioning.view(-1, 1)

        if conditioning is None:
            conditioning = torch.zeros_like(target)

        noise_probability = self.noise_cond_prob_range[0] + progress * (
            self.noise_cond_prob_range[1] - self.noise_cond_prob_range[0]
        )
        noisy_conditioning = torch.rand(batch_size, device=target.device) > 1.0 - noise_probability
        conditioning = conditioning + (
            noisy_conditioning.view(-1, 1, 1) * torch.randn_like(conditioning) * self.noise_cond_scale
        )

        ratio_r_neq_t = (
            self.ratio_r_neq_t_range[0] + progress * (self.ratio_r_neq_t_range[1] - self.ratio_r_neq_t_range[0])
            if self.mean_mode
            else 0.0
        )
        r_samples, t_samples = self.sample_r_t(target, ratio_r_neq_t=ratio_r_neq_t)
        detached_r_samples = r_samples.detach().clone()
        detached_t_samples = t_samples.detach().clone()

        noise = torch.randn_like(target)
        interpolated_states = (1 - detached_t_samples.view(-1, 1, 1)) * target + detached_t_samples.view(
            -1, 1, 1
        ) * noise
        target_velocity = noise - target

        def model_function(sample: torch.Tensor, r_timestep: torch.Tensor, t_timestep: torch.Tensor) -> torch.Tensor:
            return self.estimator(
                sample,
                mu,
                t_timestep,
                conditioning,
                delta_timestep=t_timestep - r_timestep,
            )

        if self.mean_mode:
            r_velocity = torch.zeros_like(r_samples)
            t_velocity = torch.ones_like(t_samples)
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False):
                predicted_velocity, velocity_derivative = jvp(
                    model_function,
                    (interpolated_states, r_samples, t_samples),
                    (target_velocity, r_velocity, t_velocity),
                )
            target_velocity = (
                target_velocity - (detached_t_samples - detached_r_samples).view(-1, 1, 1) * velocity_derivative
            )
        else:
            predicted_velocity = model_function(interpolated_states, r_samples, t_samples)

        losses = F.mse_loss(predicted_velocity, target_velocity.detach(), reduction="none").mean(dim=1)
        if target_mask is None:
            return losses.mean()
        weights = self.adaptive_loss_weighting(losses, target_mask.squeeze(1))
        return (weights * losses).sum() / torch.clamp(target_mask.sum(), min=1.0)


__all__ = [
    "VoxCPM2AudioVAEConfig",
    "VoxCPM2CfmConfig",
    "VoxCPM2Config",
    "VoxCPM2DiTConfig",
    "VoxCPM2EncoderConfig",
    "VoxCPM2TextConfig",
]
