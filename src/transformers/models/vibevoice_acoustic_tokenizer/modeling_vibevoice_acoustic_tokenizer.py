import copy
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import ModelOutput, auto_docstring
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_vibevoice_acoustic_tokenizer import VibeVoiceAcousticTokenizerConfig


logger = logging.get_logger(__name__)


# TODO (ebezzam) move to `src/transformers/cache_utils.py` and make similar to `DynamicCache`?
class VibeVoiceAcousticTokenizerCache:
    """
    Cache for streaming convolution, where the past convolution outputs of each layer are stored.
    Similar to KV cache for attention-based models.
    """

    def __init__(self):
        self.cache = {}  # Dict mapping (layer_idx, sample_idx) to state tensor

    def get(self, layer_idx: int, sample_indices: torch.Tensor) -> Optional[torch.Tensor]:
        """Get cached states for given layer and sample indices"""
        if sample_indices.numel() == 0:
            return None

        # Collect states for all requested samples
        states = []
        sample_indices_list = sample_indices.tolist()
        for idx in sample_indices_list:
            key = (layer_idx, idx)
            if key not in self.cache:
                return None  # If any sample is missing, return None
            states.append(self.cache[key])

        # Check if all states have the same shape for direct stacking
        first_shape = states[0].shape
        if all(state.shape == first_shape for state in states):
            return torch.stack(states, dim=0)

        # Pad to max length if shapes differ
        max_length = max(state.shape[-1] for state in states)
        padded_states = [
            F.pad(state, (max_length - state.shape[-1], 0)) if state.shape[-1] < max_length else state
            for state in states
        ]
        return torch.stack(padded_states, dim=0)

    def update(self, layer_idx: int, sample_indices: torch.Tensor, states: torch.Tensor):
        """Set cached states for given layer and sample indices"""
        sample_indices_list = sample_indices.tolist()
        for i, idx in enumerate(sample_indices_list):
            self.cache[(layer_idx, idx)] = states[i].detach()

    def set_to_zero(self, sample_indices: torch.Tensor):
        """Reset cached states for given sample indices"""
        if sample_indices.numel() == 0 or not self.cache:
            return

        sample_indices_set = set(sample_indices.tolist())
        # Remove keys (instead of zeroing them in original) for cleaner memory management
        keys_to_remove = [key for key in self.cache.keys() if key[1] in sample_indices_set]
        for key in keys_to_remove:
            del self.cache[key]

    def clear(self):
        """Clear all cached states"""
        self.cache.clear()

    @property
    def is_empty(self) -> bool:
        """Check if cache is empty"""
        return len(self.cache) == 0


class VibeVoiceAcousticTokenizerStreamingConvTranspose1d(nn.Module):
    """ConvTranspose1d with built-in handling of streaming and causal padding."""
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int, 
        stride: int = 1, 
        bias: bool = True
    ):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, bias=bias)

        # Store configuration
        self.stride = stride

        # For transposed convolution, padding calculation is different
        self.truncation_right = kernel_size - stride

        # For streaming, we need to keep track of input history
        # Transposed conv needs to see multiple input samples to produce correct output
        self.context_size = kernel_size - 1

    def forward(self, 
        x: torch.Tensor,
        past_conv_values: Optional[VibeVoiceAcousticTokenizerCache] = None,
        sample_indices: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional streaming support via cache.
        """
        batch_size, channels, time_dim = x.shape

        # Validate cache parameters if provided
        if past_conv_values is not None:
            if layer_idx is None:
                raise ValueError("layer_idx must be provided when past_conv_values is used.")
            if sample_indices is None:
                raise ValueError("sample_indices must be provided when past_conv_values is used.")
            if len(sample_indices) != batch_size:
                raise ValueError("sample_indices length must match batch size")

        # Non-streaming mode
        if past_conv_values is None:
            # Apply transposed convolution
            y = self.convtr(x)

            # Calculate and remove padding
            if self.truncation_right > 0:
                end = y.shape[-1] - self.truncation_right
                y = y[..., 0:end]
            return y

        # Streaming mode - get cached input
        cached_input = past_conv_values.get(layer_idx, sample_indices)
        if cached_input is None:
            cached_input = torch.zeros(batch_size, channels, 0, device=x.device, dtype=x.dtype)

        # Concatenate cached input with new input
        full_input = torch.cat([cached_input, x], dim=2)
        
        # Apply transposed convolution
        full_output = self.convtr(full_input)

        # Calculate and remove padding
        if self.truncation_right > 0:
            end = full_output.shape[-1] - self.truncation_right
            full_output = full_output[..., 0:end]

        # Extract output corresponding to new input
        if cached_input.shape[2] == 0:
            # First chunk - return all output
            output = full_output
        else:
            # Subsequent chunks - return only the new output
            expected_new_output = time_dim * self.stride
            if full_output.shape[2] >= expected_new_output:
                output = full_output[:, :, -expected_new_output:]
            else:
                output = full_output

        # Update cache
        if full_input.shape[2] > self.context_size:
            new_cache = full_input[:, :, -self.context_size:]
        else:
            new_cache = full_input
        past_conv_values.update(layer_idx, sample_indices, new_cache)

        return output


class VibeVoiceAcousticTokenizerRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        VibeVoiceAcousticTokenizerRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class VibeVoiceEncoderFeedForward(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, config.ffn_expansion * hidden_size, bias=config.bias)
        self.activation = ACT2FN[config.hidden_act]
        self.linear2 = nn.Linear(config.ffn_expansion * hidden_size, hidden_size, bias=config.bias)

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))


class VibeVoiceAcousticTokenizerStreamingConv1d(nn.Module):
    """Conv1d with built-in handling of streaming and causal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias
        )
        # Padding for causality: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L263C28-L263C72
        self.causal_padding = (kernel_size - 1) * dilation - (stride - 1)

    def forward(
        self,
        x: torch.Tensor,
        past_conv_values: Optional[VibeVoiceAcousticTokenizerCache] = None,
        sample_indices: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional streaming support via cache.

        Original code: https://github.com/vibevoice-community/VibeVoice/blob/63a21e2b45e908be63765bf312a9ecfb3a588315/vibevoice/modular/modular_vibevoice_tokenizer.py#L327

        Args:
            x: Input tensor [batch_size, channels, time]
            past_conv_values: `VibeVoiceAcousticTokenizerCache` object for maintaining convolution states
            sample_indices: Indices identifying each sample for cache management
            layer_idx: Layer index for cache management

        Returns:
            Output tensor
        """
        # Early return for no causal padding case
        if self.causal_padding <= 0:
            return self.conv(x)

        batch_size, channels, _ = x.shape

        # Validate cache parameters
        if past_conv_values is not None:
            if layer_idx is None:
                raise ValueError("layer_idx must be provided when past_conv_values is used.")
            if sample_indices is None:
                raise ValueError("sample_indices must be provided when past_conv_values is used.")
            if len(sample_indices) != batch_size:
                raise ValueError("sample_indices length must match batch size")

        # Get cached context
        cached_states = None
        if past_conv_values is not None:
            cached_states = past_conv_values.get(layer_idx, sample_indices)

        # Initialize with zeros if no cache exists
        if cached_states is None:
            cached_states = torch.zeros(batch_size, channels, self.causal_padding, device=x.device, dtype=x.dtype)

        # Concatenate context with input
        input_with_context = torch.cat([cached_states, x], dim=2)

        # Update cache with the last causal_padding samples from the input
        if past_conv_values is not None:
            # Use the combined input for cache update to maintain continuity
            new_cache = input_with_context[:, :, -self.causal_padding :]
            past_conv_values.update(layer_idx, sample_indices, new_cache)

        return self.conv(input_with_context)


class VibeVoiceAcousticTokenizerConvNext1dLayer(nn.Module):
    """
    ConvNeXt-like block adapted for 1D convolutions, used in VibeVoice tokenizer encoder.

    For reference, original 2D `ConvNextLayer`:
    https://github.com/huggingface/transformers/blob/e20df45bf676d80bdddb9757eeeafe6c0c81ecfa/src/transformers/models/convnext/modeling_convnext.py#L120
    """

    def __init__(self, config, hidden_size, drop_path=0.0, dilation=1, stride=1):
        super().__init__()

        self.norm = VibeVoiceAcousticTokenizerRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = VibeVoiceAcousticTokenizerRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.ffn = VibeVoiceEncoderFeedForward(config, hidden_size)
        if config.layer_scale_init_value > 0:
            self.gamma = nn.Parameter(config.layer_scale_init_value * torch.ones(hidden_size), requires_grad=True)
            self.ffn_gamma = nn.Parameter(config.layer_scale_init_value * torch.ones(hidden_size), requires_grad=True)
        else:
            self.gamma = None
            self.ffn_gamma = None

        # TODO (ebezzam) original code has option for DropPath but is never actually used (and `nn.modules.DropPath` does not exist):
        # https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L637
        # however, could be interesting feature for future versions of `ConvNext1dLayer` as the 2D version has it:
        # https://github.com/huggingface/transformers/blob/e20df45bf676d80bdddb9757eeeafe6c0c81ecfa/src/transformers/models/convnext/modeling_convnext.py#L146
        if drop_path > 0.0:
            # possible implementation (that may needed to be adapted for 1D):
            # https://github.com/huggingface/transformers/blob/e20df45bf676d80bdddb9757eeeafe6c0c81ecfa/src/transformers/models/convnext/modeling_convnext.py#L40
            raise NotImplementedError("DropPath is not implemented.")
        self.drop_path = nn.Identity()

        self.mixer = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=config.kernel_size,
            groups=hidden_size,
            bias=config.bias,
            dilation=dilation,
            stride=stride,
        )
        # Padding for causality: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L266
        self.causal_padding = (config.kernel_size - 1) * dilation - (stride - 1)

    def forward(self, x):
        residual = x
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        # Padding for causality: https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L382
        x = F.pad(x, (self.causal_padding, 0))
        x = self.mixer(x)
        if self.gamma is not None:
            x = x * self.gamma.unsqueeze(-1)
        # (ebezzam) original code (https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L653)
        # as mentioned above, drop_path is not used and the VibeVoice authors don't use the `forward` method but a custom
        # call which does `residual + x` directly (see link below), which is same as using identity
        # https://github.com/pengzhiliang/transformers/blob/6e6e60fb95ca908feb0b039483adcc009809f579/src/transformers/models/vibevoice/modular_vibevoice_tokenizer.py#L768
        x = residual + self.drop_path(x)

        # ffn
        residual = x
        x = self.ffn_norm(x.transpose(1, 2))  # [B, T, C]
        x = self.ffn(x)  # FFN expects [B, T, C]
        x = x.transpose(1, 2)  # Back to [B, C, T]
        if self.ffn_gamma is not None:
            x = x * self.ffn_gamma.unsqueeze(-1)
        # (ebezzam) see comment above
        x = residual + self.drop_path(x)
        return x


class VibeVoiceAcousticTokenizerEncoder(nn.Module):
    """
    Encoder component for the VibeVoice tokenizer that converts audio to latent representations.
    
    Args:
        config: Configuration object with model parameters
    """
    def __init__(self, config):
        super().__init__()

        # stem and intermediate downsampling conv layers
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            VibeVoiceAcousticTokenizerStreamingConv1d(
                in_channels=config.channels,
                out_channels=config.n_filters,
                kernel_size=config.kernel_size,
                bias=config.bias,
            )
        )
        for i in range(len(config.downsampling_ratios)):
            downsample_layer = VibeVoiceAcousticTokenizerStreamingConv1d(
                in_channels=config.n_filters * (2**i),
                out_channels=config.n_filters * (2 ** (i + 1)),
                kernel_size=config.downsampling_ratios[i] * 2,
                stride=config.downsampling_ratios[i],
                bias=config.bias,
            )
            self.downsample_layers.append(downsample_layer)

        # configure ConvNext1D blocks
        self.stages = nn.ModuleList()
        for i in range(len(config.depths)):
            in_ch = config.n_filters * (2**i)
            stage = nn.ModuleList(
                [VibeVoiceAcousticTokenizerConvNext1dLayer(config, hidden_size=in_ch) for _ in range(config.depths[i])]
            )
            self.stages.append(stage)

        self.head = VibeVoiceAcousticTokenizerStreamingConv1d(
            in_channels=in_ch,
            out_channels=config.hidden_size,
            kernel_size=config.kernel_size,
            bias=config.bias,
        )

    def forward(self, x, past_conv_values=None, sample_indices=None):
        for layer_idx, downsample_layer in enumerate(self.downsample_layers):
            x = downsample_layer(
                x, past_conv_values=past_conv_values, sample_indices=sample_indices, layer_idx=layer_idx
            )
            for block in self.stages[layer_idx]:
                x = block(x)
        x = self.head(x, past_conv_values=past_conv_values, sample_indices=sample_indices, layer_idx=layer_idx + 1)
        return x.permute(0, 2, 1)


class VibeVoiceAcousticTokenizerDecoder(nn.Module):
    """
    Decoder component for the VibeVoice tokenizer that converts latent representations back to audio.
    
    Args:
        config: Configuration object with model parameters
    """
    def __init__(self, config):
        super().__init__()

        # stem and upsampling layers
        self.upsample_layers = nn.ModuleList()
        self.upsample_layers.append(
            VibeVoiceAcousticTokenizerStreamingConv1d(config.hidden_size, config.n_filters * 2 ** (len(config.depths) - 1), config.kernel_size,
                    bias=config.bias)
        )
        for i in range(len(config.ratios)):
            in_ch = config.n_filters * (2 ** (len(config.depths) - 1 - i))
            out_ch = config.n_filters * (2 ** (len(config.depths) - 1 - i - 1))
            upsample_layer = VibeVoiceAcousticTokenizerStreamingConvTranspose1d(in_ch, out_ch,
                            kernel_size=config.ratios[i] * 2, stride=config.ratios[i],
                            bias=config.bias)
            self.upsample_layers.append(upsample_layer)

        # configure ConvNext1D blocks
        self.stages = nn.ModuleList()
        for i in range(len(config.depths)):
            in_ch = config.n_filters * (2 ** (len(config.depths) - 1 - i))
            stage = nn.ModuleList(
                [VibeVoiceAcousticTokenizerConvNext1dLayer(config, hidden_size=in_ch) for _ in range(config.depths[i])]
            )
            self.stages.append(stage)

        self.head = VibeVoiceAcousticTokenizerStreamingConv1d(in_ch, config.channels, kernel_size=config.kernel_size, bias=config.bias)

    def forward(self, x, past_conv_values=None, sample_indices=None):
        for layer_idx, upsample_layer in enumerate(self.upsample_layers):
            x = upsample_layer(
                x, past_conv_values=past_conv_values, sample_indices=sample_indices, layer_idx=layer_idx
            )
            for block in self.stages[layer_idx]:
                x = block(x)
        x = self.head(x, past_conv_values=past_conv_values, sample_indices=sample_indices, layer_idx=layer_idx + 1)
        return x


@dataclass
@auto_docstring
class VibeVoiceAcousticTokenizerOutput(ModelOutput):
    """
    Output of VibeVoice acoustic tokenizer model.

    Args:
        mean (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Projected latents (continuous representations for semantic tokens) at the output of the encoder.
        std
        past_conv_values (`VibeVoiceSemanticTokenizerCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            A [`VibeVoiceSemanticTokenizerCache`] instance containing cached convolution states for each layer that
            can be passed to subsequent forward calls.
    """

    mean: Optional[torch.FloatTensor] = None
    std: Optional[Union[float, torch.Tensor]] = None
    past_conv_values: Optional["VibeVoiceAcousticTokenizerCache"] = None

@dataclass
@auto_docstring
class VibeVoiceAcousticDecoderOutput(ModelOutput):
    """
    Output of VibeVoice acoustic tokenizer decoder.

    Args:
        audio (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Projected latents (continuous representations for semantic tokens) at the output of the encoder.
        past_conv_values (`VibeVoiceSemanticTokenizerCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            A [`VibeVoiceSemanticTokenizerCache`] instance containing cached convolution states for each layer that
            can be passed to subsequent forward calls.
    """

    audio: Optional[torch.FloatTensor] = None
    past_conv_values: Optional["VibeVoiceAcousticTokenizerCache"] = None
    

@dataclass
class VibeVoiceTokenizerEncoderOutput:
    """
    Output of VibeVoice tokenizer encoder, representing a Gaussian distribution with fixed variance.
    
    Args:
        mean (`torch.FloatTensor`): The mean parameters of the distribution.
        std (`float` or `torch.FloatTensor`): Fixed standard deviation value.
    """
    mean: torch.Tensor
    std: Optional[Union[float, torch.Tensor]] = None


class VibeVoiceAcousticTokenizerModel(PreTrainedModel):
    """VibeVoice speech tokenizer model combining encoder and decoder for acoustic tokens"""

    config_class = VibeVoiceAcousticTokenizerConfig
    base_model_prefix = "vibevoice_acoustic_tokenizer"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _no_split_modules = ["VibeVoiceAcousticTokenizerEncoder", "VibeVoiceAcousticTokenizerDecoder"]

    def __init__(self, config):
        super().__init__(config)

        self.register_buffer('fix_std', torch.tensor(config.fix_std), persistent=False)
        self.sample_latent = config.sample_latent

        # Create decoder config, TODO pass config directrly
        decoder_config = copy.deepcopy(config)
        decoder_config.dimension = config.hidden_size
        decoder_config.n_filters = config.n_filters
        decoder_config.ratios = config.upsampling_ratios
        decoder_config.depths = config.decoder_depths
        decoder_config.bias = config.bias
        decoder_config.layernorm_eps = config.rms_norm_eps
        decoder_config.layer_scale_init_value = config.layer_scale_init_value

        # Initialize encoder and decoder
        self.encoder = VibeVoiceAcousticTokenizerEncoder(config)
        self.decoder = VibeVoiceAcousticTokenizerDecoder(decoder_config)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for the model"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @torch.no_grad()
    def encode(self, audio, past_conv_values=None, sample_indices=None, use_cache=False):
        """Convert audio to latent representations"""
        # Input validation
        batch_size = audio.shape[0]
        if sample_indices is not None and len(sample_indices) != batch_size:
            raise ValueError(f"sample_indices length ({len(sample_indices)}) must match batch size ({batch_size})")

        if use_cache and past_conv_values is None:
            past_conv_values = VibeVoiceAcousticTokenizerCache()

        latents = self.encoder(audio, past_conv_values=past_conv_values, sample_indices=sample_indices)

        return VibeVoiceAcousticTokenizerOutput(
            mean=latents,
            std=self.fix_std,
            past_conv_values=past_conv_values if use_cache else None,
        )

    @torch.no_grad()
    def sample(self, encoder_output):
        """Sample from the encoder output distribution with a Gaussian distribution."""
        batch_size = encoder_output.mean.size(0)
        value = encoder_output.std / 0.8
        std = torch.randn(batch_size, device=encoder_output.mean.device, dtype=encoder_output.mean.dtype) * value

        while std.dim() < encoder_output.mean.dim():
            std = std.unsqueeze(-1)

        x = encoder_output.mean + std * torch.randn_like(encoder_output.mean)
        return x, std

    @torch.no_grad()
    def decode(self, latents, past_conv_values=None, sample_indices=None, use_cache=False):
        """Convert latent representations back to audio"""
        # Input validation
        batch_size = latents.shape[0]
        if sample_indices is not None and len(sample_indices) != batch_size:
            raise ValueError(f"sample_indices length ({len(sample_indices)}) must match batch size ({batch_size})")

        if use_cache and past_conv_values is None:
            past_conv_values = VibeVoiceAcousticTokenizerCache()

        latents = latents.permute(0, 2, 1)
        audio = self.decoder(latents, past_conv_values=past_conv_values, sample_indices=sample_indices)
        return VibeVoiceAcousticDecoderOutput(audio=audio, past_conv_values=past_conv_values)

    def forward(self, audio, past_conv_values=None, sample_indices=None, use_cache=False):
        """Full forward pass: encode audio to latents, then decode back to audio"""
        encoder_output = self.encode(audio, past_conv_values=past_conv_values, sample_indices=sample_indices, use_cache=use_cache)
        if self.sample_latent:
            sampled_latents, _ = self.sample(encoder_output)
        else:
            sampled_latents = encoder_output.mean
        reconstructed = self.decode(sampled_latents, past_conv_values=past_conv_values, sample_indices=sample_indices, use_cache=use_cache)
        return reconstructed, sampled_latents


__all__ = ["VibeVoiceAcousticTokenizerModel"]
