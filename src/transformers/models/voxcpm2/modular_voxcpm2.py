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

import math

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..dac.modeling_dac import Snake1d
from ..minicpm4.configuration_minicpm4 import MiniCPM4Config


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


__all__ = [
    "VoxCPM2AudioVAEConfig",
    "VoxCPM2CfmConfig",
    "VoxCPM2Config",
    "VoxCPM2DiTConfig",
    "VoxCPM2EncoderConfig",
    "VoxCPM2TextConfig",
]
