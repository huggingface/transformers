import math
import typing as tp
from functools import partial
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...models.auto import AutoModel
from ...utils import logging
from ...modeling_utils import PreTrainedModel
from ...activations import ACT2FN

from .configuration_vibevoice import VibeVoiceAcousticTokenizerConfig, VibeVoiceSemanticTokenizerConfig

logger = logging.get_logger(__name__)

# Normalization modules
class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """
    def __init__(self, normalized_shape: tp.Union[int, tp.List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = x.transpose(1, 2)  # b ... t -> b t ...
        x = nn.functional.layer_norm(x.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).type_as(x) 
        x = x.transpose(1, 2)  # b t ... -> b ... t
        return x
    
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True, weight_shape=None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            weight_shape = (dim,) if weight_shape is None else weight_shape
            self.weight = nn.Parameter(torch.ones(weight_shape))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

class ConvRMSNorm(RMSNorm):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True, weight_shape=None):
        super().__init__(dim, eps, elementwise_affine, weight_shape)

    def forward(self, x):
        x = x.transpose(1, 2)  # b ... t -> b t ...
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight

        output = output.transpose(1, 2)  # b t ... -> b ... t
        return output

# Convolutional layers and utilities
CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm',
                                'time_layer_norm', 'layer_norm', 'time_group_norm'])


def apply_parametrization_norm(module: nn.Module, norm: str = 'none') -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return nn.utils.weight_norm(module)
    elif norm == 'spectral_norm':
        return nn.utils.spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


def get_norm_module(module: nn.Module, causal: bool = False, norm: str = 'none', **norm_kwargs) -> nn.Module:
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'layer_norm':
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                padding_total: int = 0) -> int:
    """Calculate extra padding needed for convolution to have the same output length"""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'zero', value: float = 0.):
    """Pad 1D input with handling for small inputs in reflect mode"""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left: end]


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv"""
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose1d(nn.Module):
    """Wrapper around ConvTranspose1d and normalization applied to this conv"""
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


class VibeVoiceTokenizerStreamingCache:
    """Cache for streaming convolution, similar to KV cache in attention"""
    def __init__(self):
        self.cache = {}  # Dict mapping (layer_id, sample_idx) to state tensor
        
    def get(self, layer_id: str, sample_indices: torch.Tensor) -> Optional[torch.Tensor]:
        """Get cached states for given layer and sample indices"""
        states = []
        max_length = 0
        
        # First pass: collect states and find max length
        for idx in sample_indices.tolist():
            key = (layer_id, idx)
            if key not in self.cache:
                return None  # If any sample is missing, return None
            state = self.cache[key]
            states.append(state)
            max_length = max(max_length, state.shape[-1])
        
        # Second pass: pad states to max length if needed
        if len(states) > 0 and states[0].dim() >= 2:
            padded_states = []
            for state in states:
                if state.shape[-1] < max_length:
                    # Pad on the time dimension (last dimension)
                    pad_size = max_length - state.shape[-1]
                    # Pad with zeros on the LEFT to align the most recent samples
                    padded_state = F.pad(state, (pad_size, 0), mode='constant', value=0)
                    padded_states.append(padded_state)
                else:
                    padded_states.append(state)
            return torch.stack(padded_states, dim=0)
        else:
            return torch.stack(states, dim=0)
    
    def set(self, layer_id: str, sample_indices: torch.Tensor, states: torch.Tensor):
        """Set cached states for given layer and sample indices"""
        for i, idx in enumerate(sample_indices.tolist()):
            key = (layer_id, idx)
            self.cache[key] = states[i].detach()

    def set_to_zero(self, sample_indices: torch.Tensor):
        """Set all cached states to zero for given sample indices"""
        for key in list(self.cache.keys()):
            layer_id, sample_idx = key
            if sample_idx in sample_indices.tolist():
                # Create zero tensor with same shape and dtype as cached tensor
                cached_tensor = self.cache[key]
                self.cache[key] = torch.zeros_like(cached_tensor)
                
    def clear(self, layer_id: Optional[str] = None, sample_indices: Optional[torch.Tensor] = None):
        """Clear cache for specific layer/samples or everything"""
        if layer_id is None and sample_indices is None:
            self.cache.clear()
        elif layer_id is not None and sample_indices is None:
            # Clear all samples for a specific layer
            keys_to_remove = [k for k in self.cache.keys() if k[0] == layer_id]
            for k in keys_to_remove:
                del self.cache[k]
        elif layer_id is not None and sample_indices is not None:
            # Clear specific samples for a specific layer
            for idx in sample_indices.tolist():
                key = (layer_id, idx)
                self.cache.pop(key, None)

class SConv1d(nn.Module):
    """Conv1d with built-in handling of asymmetric or causal padding and normalization."""
    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: int, stride: int = 1, dilation: int = 1,
                groups: int = 1, bias: bool = True, causal: bool = False,
                norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {},
                pad_mode: str = 'reflect'):
        super().__init__()
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride,
                            dilation=dilation, groups=groups, bias=bias, causal=causal,
                            norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode
        
        # Store configuration
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # For causal convolution, we need to maintain kernel_size - 1 samples as context
        # need to check use which context_size is more suitable
        # self.context_size = (kernel_size - 1) * dilation
        self.context_size = (kernel_size - 1) * dilation - (stride - 1)
        
        # For non-streaming mode, calculate padding
        self.padding_total = (kernel_size - 1) * dilation - (stride - 1)
        
        # Create a unique layer ID for cache management
        self._layer_id = None
                  
    @property
    def layer_id(self):
        if self._layer_id is None:
            self._layer_id = f"sconv1d_{id(self)}"
        return self._layer_id
        
    def forward(self, x: torch.Tensor, 
                cache: Optional[VibeVoiceTokenizerStreamingCache] = None,
                sample_indices: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                debug: bool = False) -> torch.Tensor:
        """
        Forward pass with optional streaming support via cache.
        
        Args:
            x: Input tensor [batch_size, channels, time]
            cache: VibeVoiceTokenizerStreamingCache object for maintaining states
            sample_indices: Indices identifying each sample for cache management
            use_cache: Whether to use cached states for streaming
            debug: Whether to print debug information
            
        Returns:
            Output tensor
        """
        B, C, T = x.shape
        
        # Non-streaming mode
        if not use_cache or cache is None:
            return self._forward_non_streaming(x, debug=debug)
        
        # Streaming mode
        assert self.causal, "Streaming mode is only supported for causal convolutions"
        assert sample_indices is not None, "sample_indices must be provided for streaming mode"
        assert len(sample_indices) == B, "sample_indices must match batch size"
        
        return self._forward_streaming(x, cache, sample_indices, debug)
    
    def _forward_streaming(self, x: torch.Tensor, 
                          cache: VibeVoiceTokenizerStreamingCache,
                          sample_indices: torch.Tensor,
                          debug: bool = False) -> torch.Tensor:
        """Streaming forward pass with cache operations kept separate from compiled code"""
        B, C, T = x.shape
        
        # Cache operations (not compiled)
        cached_states = cache.get(self.layer_id, sample_indices)
        
        if cached_states is None:
            # First chunk - initialize with zeros for context
            if self.context_size > 0:
                cached_states = torch.zeros(B, C, self.context_size, device=x.device, dtype=x.dtype)
                if debug:
                    print(f"[DEBUG] Initialized cache with shape: {cached_states.shape}, context_size={self.context_size}")
            else:
                cached_states = torch.zeros(B, C, 0, device=x.device, dtype=x.dtype)
                if debug:
                    print(f"[DEBUG] No context needed (kernel_size=stride)")
        
        # Concatenate cached states with input
        if cached_states.shape[2] > 0:
            input_with_context = torch.cat([cached_states, x], dim=2)
        else:
            input_with_context = x
            
        if debug:
            print(f"[DEBUG] Input shape: {x.shape}, Cache shape: {cached_states.shape}, Combined: {input_with_context.shape}")
        
        # Apply convolution directly - no extra padding in streaming mode
        # The conv layer will handle its own padding internally
        output = self.conv(input_with_context)

        if debug:
            print(f"[DEBUG] Output shape: {output.shape}")
        
        # Update cache for next chunk
        if self.context_size > 0:
            # Calculate how many samples to keep
            total_input_length = input_with_context.shape[2]
            
            # Keep the last context_size samples
            if total_input_length >= self.context_size:
                new_cache_start = total_input_length - self.context_size
                new_cache = input_with_context[:, :, new_cache_start:]
            else:
                # If we have less than context_size samples, keep everything
                new_cache = input_with_context
                
            if debug:
                print(f"[DEBUG] New cache shape: {new_cache.shape}")
                
            cache.set(self.layer_id, sample_indices, new_cache)
        
        return output
    
    def _forward_non_streaming(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """Standard forward pass without streaming"""
        B, C, T = x.shape
        kernel_size = self.kernel_size
        stride = self.stride
        dilation = self.dilation
        padding_total = self.padding_total
        
        # Compute extra padding for stride alignment
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        
        if debug:
            print(f"[DEBUG NON-STREAMING] Input shape: {x.shape}, padding_total={padding_total}, extra_padding={extra_padding}")
        
        if self.causal:
            # Left padding for causal
            if self.pad_mode == 'constant':
                x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode, value=0)
            else:
                x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Symmetric padding for non-causal
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        
        if debug:
            print(f"[DEBUG NON-STREAMING] After padding: {x.shape}")
            
        output = self.conv(x)
        
        if debug:
            print(f"[DEBUG NON-STREAMING] Output shape: {output.shape}")
        
        return output


class SConvTranspose1d(nn.Module):
    """ConvTranspose1d with built-in handling of asymmetric or causal padding and normalization."""
    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: int, stride: int = 1, causal: bool = False,
                norm: str = 'none', trim_right_ratio: float = 1.,
                norm_kwargs: tp.Dict[str, tp.Any] = {}, bias: bool = True):
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels, out_channels, kernel_size, stride,
                                        causal=causal, norm=norm, norm_kwargs=norm_kwargs, bias=bias)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1., \
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0. and self.trim_right_ratio <= 1.

        # Store configuration
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # For transposed convolution, padding calculation is different
        self.padding_total = kernel_size - stride
        
        # For streaming, we need to keep track of input history
        # Transposed conv needs to see multiple input samples to produce correct output
        self.context_size = kernel_size - 1
        
        # Create a unique layer ID for cache management
        self._layer_id = None

    @property
    def layer_id(self):
        if self._layer_id is None:
            self._layer_id = f"sconvtr1d_{id(self)}"
        return self._layer_id
    
    def forward(self, x: torch.Tensor,
                cache: Optional[VibeVoiceTokenizerStreamingCache] = None,
                sample_indices: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                debug: bool = False) -> torch.Tensor:
        """
        Forward pass with optional streaming support via cache.
        """
        B, C, T = x.shape
        
        # Non-streaming mode
        if not use_cache or cache is None:
            return self._forward_non_streaming(x, debug=debug)
        
        # Streaming mode
        assert sample_indices is not None, "sample_indices must be provided for streaming mode"
        assert len(sample_indices) == B, "sample_indices must match batch size"
        
        return self._forward_streaming(x, cache, sample_indices, debug)
    
    def _forward_streaming(self, x: torch.Tensor,
                          cache: VibeVoiceTokenizerStreamingCache,
                          sample_indices: torch.Tensor,
                          debug: bool = False) -> torch.Tensor:
        """Streaming forward pass with cache operations kept separate from compiled code"""
        B, C, T = x.shape
        
        # Cache operations (not compiled)
        cached_input = cache.get(self.layer_id, sample_indices)
        
        if cached_input is None:
            # First chunk - no history yet
            cached_input = torch.zeros(B, C, 0, device=x.device, dtype=x.dtype)
            if debug:
                print(f"[DEBUG] Initialized empty cache for transposed conv")
        
        # Concatenate cached input with new input
        full_input = torch.cat([cached_input, x], dim=2)
        
        if debug:
            print(f"[DEBUG] Input shape: {x.shape}, Cache shape: {cached_input.shape}, Combined: {full_input.shape}")
        
        # First chunk or debug mode - use uncompiled version
        full_output = self.convtr(full_input)
        
        if debug:
            print(f"[DEBUG] Full transposed conv output shape: {full_output.shape}")
        
        # Calculate padding to remove
        if self.causal:
            padding_right = math.ceil(self.padding_total * self.trim_right_ratio)
            padding_left = self.padding_total - padding_right
        else:
            padding_right = self.padding_total // 2
            padding_left = self.padding_total - padding_right
        
        # Remove padding
        if padding_left + padding_right > 0:
            full_output = unpad1d(full_output, (padding_left, padding_right))
        
        if debug:
            print(f"[DEBUG] After unpadding: {full_output.shape}")
        
        # Determine which part of the output corresponds to the new input
        if cached_input.shape[2] == 0:
            # First chunk - return all output
            output = full_output
        else:
            # Subsequent chunks - return only the new output
            expected_new_output = T * self.stride
            
            # Take the last expected_new_output samples
            if full_output.shape[2] >= expected_new_output:
                output = full_output[:, :, -expected_new_output:]
            else:
                output = full_output
        
        if debug:
            print(f"[DEBUG] Final streaming output shape: {output.shape}")
        
        # Update cache
        if full_input.shape[2] > self.context_size:
            new_cache = full_input[:, :, -self.context_size:]
        else:
            new_cache = full_input
        
        if debug:
            print(f"[DEBUG] New cache shape: {new_cache.shape}")
            
        cache.set(self.layer_id, sample_indices, new_cache)
        
        return output
    
    def _forward_non_streaming(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """Standard forward pass without streaming"""
        if debug:
            print(f"[DEBUG NON-STREAMING] Input shape: {x.shape}")
        
        # Apply transposed convolution
        y = self.convtr(x)
        
        if debug:
            print(f"[DEBUG NON-STREAMING] After transposed conv: {y.shape}")
        
        # Calculate and remove padding
        if self.causal:
            padding_right = math.ceil(self.padding_total * self.trim_right_ratio)
            padding_left = self.padding_total - padding_right
        else:
            padding_right = self.padding_total // 2
            padding_left = self.padding_total - padding_right
        
        if padding_left + padding_right > 0:
            y = unpad1d(y, (padding_left, padding_right))
        
        if debug:
            print(f"[DEBUG NON-STREAMING] Final output shape: {y.shape}")
            
        return y
    
# FFN 
class FFN(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        bias=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(self.embed_dim, ffn_dim, bias=bias) 
        self.gelu = ACT2FN["gelu"]
        self.linear2 = nn.Linear(ffn_dim, self.embed_dim, bias=bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class Convlayer(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            dilation=1, 
            groups=1, 
            bias=True, 
            pad_mode='zeros', 
            norm='weight_norm', 
            causal=True, 
        ):
        super().__init__()
        self.conv = SConv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, 
                           groups=groups, bias=bias, pad_mode=pad_mode, norm=norm, causal=causal)

    def forward(self, x):
        return self.conv(x)

class Block1D(nn.Module):
    def __init__(self, dim, kernel_size=7, drop_path=0., mixer_layer='conv',  
                layer_scale_init_value=1e-6, **kwargs):
        super().__init__()
        
        if kwargs.get('layernorm', 'LN') == 'LN':
            self.norm = ConvLayerNorm(dim, eps=kwargs.get('eps', 1e-6))
            self.ffn_norm = ConvLayerNorm(dim, eps=kwargs.get('eps', 1e-6))               
        elif kwargs.get('layernorm', 'RMSNorm') == 'RMSNorm':
            self.norm = ConvRMSNorm(dim, eps=kwargs.get('eps', 1e-6))
            self.ffn_norm = ConvRMSNorm(dim, eps=kwargs.get('eps', 1e-6))

        if mixer_layer == 'conv':
            self.mixer = Convlayer(dim, dim, groups=kwargs.get('groups', 1),
                                kernel_size=kernel_size, 
                                pad_mode=kwargs.get('pad_mode', 'reflect'), 
                                norm=kwargs.get('norm', 'none'), 
                                causal=kwargs.get('causal', True), 
                                bias=kwargs.get('bias', True),
                                )
        elif mixer_layer == 'depthwise_conv':
            self.mixer = Convlayer(dim, dim, groups=dim,
                                kernel_size=kernel_size, 
                                pad_mode=kwargs.get('pad_mode', 'reflect'), 
                                norm=kwargs.get('norm', 'none'), 
                                causal=kwargs.get('causal', True), 
                                bias=kwargs.get('bias', True),
                                )
        else:
            raise ValueError(f"Unsupported mixer layer: {mixer_layer}")
        
        self.ffn = FFN(
            dim, 
            kwargs.get('ffn_expansion', 4) * dim, 
            bias=kwargs.get('bias', False),
        )
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.modules.DropPath(drop_path)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.ffn_gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma = None
            self.ffn_gamma = None

    def forward(self, x):
        # mixer
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        if self.gamma is not None:
            x = x * self.gamma.unsqueeze(-1)
        x = residual + self.drop_path(x)

        # ffn
        residual = x
        x = self.ffn_norm(x)
        x = x.permute(0, 2, 1)
        x = self.ffn(x)
        x = x.permute(0, 2, 1)
        if self.ffn_gamma is not None:
            x = x * self.ffn_gamma.unsqueeze(-1)
        x = residual + self.drop_path(x)

        return x


class TokenizerEncoder(nn.Module):
    """
    Encoder component for the VibeVoice tokenizer that converts audio to latent representations.
    
    Args:
        config: Configuration object with model parameters
    """
    def __init__(self, config):
        super().__init__()
        
        # Extract parameters from config
        self.channels = config.channels
        self.dimension = config.dimension
        self.n_filters = config.n_filters
        self.ratios = list(reversed(config.ratios))
        self.depths = config.depths
        self.n_residual_layers = getattr(config, "n_residual_layers", 1)
        self.hop_length = np.prod(self.ratios)
        self.causal = config.causal
        
        # Additional config parameters with defaults
        kernel_size = getattr(config, "kernel_size", 7)
        last_kernel_size = getattr(config, "last_kernel_size", 7)
        norm = getattr(config, "norm", "none")
        norm_params = getattr(config, "norm_params", {})
        pad_mode = getattr(config, "pad_mode", "reflect")
        bias = getattr(config, "bias", True)
        layernorm = getattr(config, "layernorm", "LN")
        layernorm_eps = getattr(config, "layernorm_eps", 1e-6)
        layernorm_elementwise_affine = getattr(config, "layernorm_elementwise_affine", True)
        drop_path_rate = getattr(config, "drop_path_rate", 0.0)
        mixer_layer = getattr(config, "mixer_layer", "conv")
        layer_scale_init_value = getattr(config, "layer_scale_init_value", 0)
        disable_last_norm = getattr(config, "disable_last_norm", False)
        
        # determine the norm type based on layernorm
        if layernorm == 'LN':
            norm_type = ConvLayerNorm
        elif layernorm == 'RMSNorm':
            norm_type = partial(ConvRMSNorm, elementwise_affine=layernorm_elementwise_affine)
        else:
            raise ValueError(f"Unsupported norm type: {layernorm}")
        
        # stem and intermediate downsampling conv layers
        stem = nn.Sequential(
                SConv1d(self.channels, self.n_filters, kernel_size, norm=norm, norm_kwargs=norm_params, causal=self.causal, pad_mode=pad_mode, bias=bias),
            )
        
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(stem)
        for i in range(len(self.ratios)):
            in_ch = self.n_filters * (2 ** i)
            out_ch = self.n_filters * (2 ** (i + 1))
            downsample_layer = nn.Sequential(
                SConv1d(in_ch, out_ch, kernel_size=self.ratios[i] * 2, stride=self.ratios[i], causal=self.causal, pad_mode=pad_mode, norm=norm, bias=bias)
            )
            self.downsample_layers.append(downsample_layer)

        # configure the transformer blocks
        layer_type = partial(
            Block1D,
            mixer_layer=mixer_layer,
            layernorm=layernorm,
            eps=layernorm_eps,
            causal=self.causal,
            pad_mode=pad_mode,
            norm=norm,
            bias=bias,
            layer_scale_init_value=layer_scale_init_value,
        )
        
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))] 
        cur = 0

        for i in range(len(self.depths)):
            in_ch = self.n_filters * (2 ** i)
            stage = nn.Sequential(
                *[layer_type(dim=in_ch, drop_path=dp_rates[cur + j]) for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]
        
        if not disable_last_norm:
            self.norm = norm_type(in_ch, eps=layernorm_eps)
        else:
            self.norm = nn.Identity()
        self.head = SConv1d(in_ch, self.dimension, kernel_size=last_kernel_size, causal=self.causal, pad_mode=pad_mode, norm=norm, bias=bias)

    def forward_features(self, x, cache=None, sample_indices=None, use_cache=False, debug=False):
        for i in range(len(self.depths)):
            # Apply downsampling
            for layer in self.downsample_layers[i]:
                if isinstance(layer, SConv1d):
                    x = layer(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
                else:
                    x = layer(x)
            
            # Apply stage (Block1D contains Convlayer which contains SConv1d)
            for block in self.stages[i]:
                if hasattr(block, 'mixer') and hasattr(block.mixer, 'conv') and isinstance(block.mixer.conv, SConv1d):
                    # Block1D forward with cache support
                    residual = x
                    x = block.norm(x)
                    x = block.mixer.conv(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
                    if block.gamma is not None:
                        x = x * block.gamma.unsqueeze(-1)
                    x = residual + x
                    
                    # FFN part
                    residual = x
                    x = block.ffn_norm(x)
                    x = x.permute(0, 2, 1)
                    x = block.ffn(x)
                    x = x.permute(0, 2, 1)
                    if block.ffn_gamma is not None:
                        x = x * block.ffn_gamma.unsqueeze(-1)
                    x = residual + x
                else:
                    x = block(x)

        return self.norm(x)

    def forward(self, x, cache=None, sample_indices=None, use_cache=False, debug=False):
        x = self.forward_features(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
        x = self.head(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
        return x


class TokenizerDecoder(nn.Module):
    """
    Decoder component for the VibeVoice tokenizer that converts latent representations back to audio.
    
    Args:
        config: Configuration object with model parameters
    """
    def __init__(self, config):
        super().__init__()
        
        # Extract parameters from config
        self.dimension = config.dimension
        self.channels = config.channels
        self.n_filters = config.n_filters
        self.ratios = config.ratios
        
        # IMPORTANT CHANGE: Don't reverse depths again since they're already reversed in VibeVoiceAcousticTokenizerModel
        self.depths = config.depths  # Changed from list(reversed(config.depths))
        
        self.n_residual_layers = getattr(config, "n_residual_layers", 1)
        self.hop_length = np.prod(self.ratios)
        self.causal = config.causal
        
        # Additional config parameters with defaults
        kernel_size = getattr(config, "kernel_size", 7)
        last_kernel_size = getattr(config, "last_kernel_size", 7)
        norm = getattr(config, "norm", "none")
        norm_params = getattr(config, "norm_params", {})
        pad_mode = getattr(config, "pad_mode", "reflect")
        bias = getattr(config, "bias", True)
        layernorm = getattr(config, "layernorm", "LN")
        layernorm_eps = getattr(config, "layernorm_eps", 1e-6)
        trim_right_ratio = getattr(config, "trim_right_ratio", 1.0)
        layernorm_elementwise_affine = getattr(config, "layernorm_elementwise_affine", True)
        drop_path_rate = getattr(config, "drop_path_rate", 0.0)
        mixer_layer = getattr(config, "mixer_layer", "conv")
        layer_scale_init_value = getattr(config, "layer_scale_init_value", 0)
        disable_last_norm = getattr(config, "disable_last_norm", False)

        # determine the norm type based on layernorm
        if layernorm == 'LN':
            norm_type = ConvLayerNorm
        elif layernorm == 'RMSNorm':
            norm_type = partial(ConvRMSNorm, elementwise_affine=layernorm_elementwise_affine)
        else:
            raise ValueError(f"Unsupported norm type: {layernorm}")
        
        # stem and upsampling layers
        stem = nn.Sequential(
                SConv1d(self.dimension, self.n_filters * 2 ** (len(self.depths) - 1), kernel_size, norm=norm, 
                        norm_kwargs=norm_params, causal=self.causal, pad_mode=pad_mode, bias=bias),
            )
        
        self.upsample_layers = nn.ModuleList()
        self.upsample_layers.append(stem)
        for i in range(len(self.ratios)):
            in_ch = self.n_filters * (2 ** (len(self.depths) - 1 - i))
            out_ch = self.n_filters * (2 ** (len(self.depths) - 1 - i - 1))
            upsample_layer = nn.Sequential(
                SConvTranspose1d(in_ch, out_ch,
                                kernel_size=self.ratios[i] * 2, stride=self.ratios[i],
                                norm=norm, norm_kwargs=norm_params, bias=bias,
                                causal=self.causal, trim_right_ratio=trim_right_ratio),
            )
            self.upsample_layers.append(upsample_layer)

        # configure transformer blocks
        layer_type = partial(
            Block1D,
            mixer_layer=mixer_layer,
            layernorm=layernorm,
            eps=layernorm_eps,
            causal=self.causal,
            pad_mode=pad_mode,
            norm=norm,
            bias=bias,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))] 
        cur = 0
        
        # Create stages in the same order as the original model
        for i in range(len(self.depths)):
            in_ch = self.n_filters * (2 ** (len(self.depths) - 1 - i))
            stage = nn.Sequential(
                *[layer_type(dim=in_ch, drop_path=dp_rates[cur + j]) for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]

        if not disable_last_norm:
            self.norm = norm_type(in_ch, eps=layernorm_eps)
        else:
            self.norm = nn.Identity()
        self.head = SConv1d(in_ch, self.channels, kernel_size=last_kernel_size, causal=self.causal, pad_mode=pad_mode, norm=norm, bias=bias)

    def forward_features(self, x, cache=None, sample_indices=None, use_cache=False, debug=False):
        for i in range(len(self.depths)):
            # Apply upsampling
            for layer in self.upsample_layers[i]:
                if isinstance(layer, (SConv1d, SConvTranspose1d)):
                    x = layer(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
                else:
                    x = layer(x)
            
            # Apply stage (Block1D contains Convlayer which contains SConv1d)
            for block in self.stages[i]:
                if hasattr(block, 'mixer') and hasattr(block.mixer, 'conv') and isinstance(block.mixer.conv, SConv1d):
                    # Block1D forward with cache support
                    residual = x
                    x = block.norm(x)
                    x = block.mixer.conv(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
                    if block.gamma is not None:
                        x = x * block.gamma.unsqueeze(-1)
                    x = residual + x
                    
                    # FFN part
                    residual = x
                    x = block.ffn_norm(x)
                    x = x.permute(0, 2, 1)
                    x = block.ffn(x)
                    x = x.permute(0, 2, 1)
                    if block.ffn_gamma is not None:
                        x = x * block.ffn_gamma.unsqueeze(-1)
                    x = residual + x
                else:
                    x = block(x)

        return self.norm(x)
    
    def forward(self, x, cache=None, sample_indices=None, use_cache=False, debug=False):
        x = self.forward_features(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
        x = self.head(x, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
        return x
    

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
    
    def sample(self, dist_type='fix'):
        """
        Sample from the distribution.
        
        Args:
            dist_type (`str`): Sampling method, either 'fix' or 'gaussian'.
                
        Returns:
            `torch.FloatTensor`: Sampled values.
            `torch.FloatTensor` (optional): Standard deviation used (only when dist_type='gaussian').
        """
        if dist_type == 'fix':
            x = self.mean + self.std * torch.randn_like(self.mean)
            return x, self.std
        elif dist_type == 'gaussian':
            batch_size = self.mean.size(0)
            value = self.std / 0.8
            std = torch.randn(batch_size, device=self.mean.device, dtype=self.mean.dtype) * value

            while std.dim() < self.mean.dim():
                std = std.unsqueeze(-1)

            x = self.mean + std * torch.randn_like(self.mean)
            return x, std
        else:
            return self.mean, self.std

    def kl(self):
        """Compute KL divergence between this distribution and a standard normal."""
        target = torch.zeros_like(self.mean)
        return F.mse_loss(self.mean, target, reduction='none')

    def mode(self):
        """Return the distribution mode (which is the mean for Gaussian)."""
        return self.mean
    
class VibeVoiceAcousticTokenizerModel(PreTrainedModel):
    """VibeVoice speech tokenizer model combining encoder and decoder for acoustic tokens"""
    
    config_class = VibeVoiceAcousticTokenizerConfig
    base_model_prefix = "vibevoice_acoustic_tokenizer"
    _supports_flash_attn_2 = True  
    _supports_sdpa = True  
    _no_split_modules = ["TokenizerEncoder", "TokenizerDecoder"]

    def __init__(self, config):
        super().__init__(config)
        
        self.register_buffer('fix_std', torch.tensor(config.fix_std), persistent=False)
        self.std_dist_type = getattr(config, "std_dist_type", "fix")
        
        # Parse encoder depths
        if isinstance(config.encoder_depths, str):
            encoder_depths = [int(d) for d in config.encoder_depths.split('-')]
        else:
            encoder_depths = config.encoder_depths
            
        # Parse decoder depths if provided
        if config.decoder_depths is not None and isinstance(config.decoder_depths, str):
            decoder_depths = [int(d) for d in config.decoder_depths.split('-')]
        else:
            # Default: use reversed encoder depths if decoder_depths is None
            decoder_depths = list(reversed(encoder_depths))
        
        # Create encoder config
        encoder_config = copy.deepcopy(config)
        encoder_config.dimension = config.vae_dim
        encoder_config.n_filters = config.encoder_n_filters
        encoder_config.ratios = config.encoder_ratios
        encoder_config.depths = encoder_depths
        encoder_config.norm = config.conv_norm
        encoder_config.pad_mode = config.pad_mode
        encoder_config.bias = config.conv_bias
        encoder_config.layernorm_eps = config.layernorm_eps
        encoder_config.layernorm_elementwise_affine = config.layernorm_elementwise_affine
        encoder_config.mixer_layer = config.mixer_layer
        encoder_config.layer_scale_init_value = config.layer_scale_init_value
        encoder_config.disable_last_norm = config.disable_last_norm
        
        # Create decoder config
        decoder_config = copy.deepcopy(config)
        decoder_config.dimension = config.vae_dim
        decoder_config.n_filters = config.decoder_n_filters
        decoder_config.ratios = config.decoder_ratios
        decoder_config.depths = decoder_depths
        decoder_config.norm = config.conv_norm
        decoder_config.pad_mode = config.pad_mode
        decoder_config.bias = config.conv_bias
        decoder_config.layernorm_eps = config.layernorm_eps
        decoder_config.layernorm_elementwise_affine = config.layernorm_elementwise_affine
        decoder_config.mixer_layer = config.mixer_layer
        decoder_config.layer_scale_init_value = config.layer_scale_init_value
        decoder_config.disable_last_norm = config.disable_last_norm
        
        # Initialize encoder and decoder
        self.encoder = TokenizerEncoder(encoder_config)
        self.decoder = TokenizerDecoder(decoder_config)
        
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
    def encode(self, audio, cache=None, sample_indices=None, use_cache=False, debug=False):
        """Convert audio to latent representations"""
        latents = self.encoder(audio, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
        return VibeVoiceTokenizerEncoderOutput(mean=latents.permute(0, 2, 1), std=self.fix_std)
    
    @torch.no_grad()
    def sampling(self, encoder_output, dist_type=None):
        """Sample from the encoder output distribution"""
        dist_type = dist_type or self.std_dist_type
    
        if dist_type == 'fix':
            return encoder_output.sample(dist_type='fix')
        elif dist_type == 'gaussian':
            return encoder_output.sample(dist_type='gaussian')
        else:
            raise ValueError(f"Unsupported dist_type: {dist_type}, expected 'fix' or 'gaussian'")
    
    @torch.no_grad()
    def decode(self, latents, cache=None, sample_indices=None, use_cache=False, debug=False):
        """Convert latent representations back to audio"""
        if latents.shape[1] == self.config.vae_dim:
            pass
        else:
            latents = latents.permute(0, 2, 1)

        audio = self.decoder(latents, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
        return audio

    def forward(self, audio, cache=None, sample_indices=None, use_cache=False, debug=False):
        """Full forward pass: encode audio to latents, then decode back to audio"""
        encoder_output = self.encode(audio, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
        sampled_latents, _ = self.sampling(encoder_output)
        reconstructed = self.decode(sampled_latents, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
        return reconstructed, sampled_latents


class VibeVoiceSemanticTokenizerModel(PreTrainedModel):
    """VibeVoice speech tokenizer model with only encoder for semantic tokens"""
    
    config_class = VibeVoiceSemanticTokenizerConfig
    base_model_prefix = "vibevoice_semantic_tokenizer"
    _supports_flash_attn_2 = True  
    _supports_sdpa = True  
    _no_split_modules = ["TokenizerEncoder"]
    
    def __init__(self, config):
        super().__init__(config)
        
        # Parse encoder depths
        if isinstance(config.encoder_depths, str):
            encoder_depths = [int(d) for d in config.encoder_depths.split('-')]
        else:
            encoder_depths = config.encoder_depths
        
        # Create encoder config
        encoder_config = copy.deepcopy(config)
        encoder_config.dimension = config.vae_dim
        encoder_config.n_filters = config.encoder_n_filters
        encoder_config.ratios = config.encoder_ratios
        encoder_config.depths = encoder_depths
        encoder_config.norm = config.conv_norm
        encoder_config.pad_mode = config.pad_mode
        encoder_config.bias = config.conv_bias
        encoder_config.layernorm_eps = config.layernorm_eps
        encoder_config.layernorm_elementwise_affine = config.layernorm_elementwise_affine
        encoder_config.mixer_layer = config.mixer_layer
        encoder_config.layer_scale_init_value = config.layer_scale_init_value
        encoder_config.disable_last_norm = config.disable_last_norm
        
        # Initialize encoder and decoder
        self.encoder = TokenizerEncoder(encoder_config)
        
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
    def encode(self, audio, cache=None, sample_indices=None, use_cache=False, debug=False):
        """Convert audio to latent representations"""
        latents = self.encoder(audio, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
        return VibeVoiceTokenizerEncoderOutput(mean=latents.permute(0, 2, 1))
    
    @torch.no_grad()
    def sampling(self, encoder_output, dist_type=None):
        """Sample from the encoder output distribution"""
        return encoder_output.sample(dist_type='none')

    def forward(self, audio, cache=None, sample_indices=None, use_cache=False, debug=False):
        """Full forward pass: encode audio to latents, then decode back to audio"""
        encoder_output = self.encode(audio, cache=cache, sample_indices=sample_indices, use_cache=use_cache, debug=debug)
        sampled_latents, _ = self.sampling(encoder_output, dist_type='none')
        return None, sampled_latents

AutoModel.register(VibeVoiceAcousticTokenizerConfig, VibeVoiceAcousticTokenizerModel)
AutoModel.register(VibeVoiceSemanticTokenizerConfig, VibeVoiceSemanticTokenizerModel)

__all__ = [
    "VibeVoiceTokenizerStreamingCache",
    "VibeVoiceAcousticTokenizerModel",
    "VibeVoiceSemanticTokenizerModel",
]