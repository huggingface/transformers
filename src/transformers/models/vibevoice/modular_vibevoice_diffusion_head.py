import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...models.auto import AutoModel
from ...modeling_utils import PreTrainedModel
from ...activations import ACT2FN
from ...utils import logging

from .configuration_vibevoice import VibeVoiceDiffusionHeadConfig


logger = logging.get_logger(__name__)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
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
    
def modulate(x, shift, scale):
    """Apply modulation to input tensor."""
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    
    Args:
        hidden_size (`int`): Size of the output embedding
        frequency_embedding_size (`int`, optional): Size of the intermediate frequency embedding
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            # nn.SiLU(),
            ACT2FN['silu'],
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t (`torch.Tensor`): A 1-D Tensor of N indices, one per batch element.
                            These may be fractional.
            dim (`int`): The dimension of the output.
            max_period (`int`, optional): Controls the minimum frequency of the embeddings.
            
        Returns:
            `torch.Tensor`: An [N, D] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(t.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FeedForwardNetwork(nn.Module):
    """
    Standard feed-forward network with SwiGLU activation.
    
    Args:
        embed_dim (`int`): Input dimension
        ffn_dim (`int`): Hidden dimension
    """
    def __init__(
        self,
        embed_dim,
        ffn_dim,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate_proj = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, self.embed_dim, bias=False)
        self.act_fn = ACT2FN['silu']  # Using SiLU as the activation function

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # SwiGLU activation
        # gate = F.silu(gate)
        gate = self.act_fn(gate)
        return self.down_proj(gate * up)

    
class HeadLayer(nn.Module):
    """
    A layer in the diffusion head.
    
    Args:
        embed_dim (`int`): Input dimension
        ffn_dim (`int`): Hidden dimension
        cond_dim (`int`): Condition embedding dimension
        norm_eps (`float`, optional): Epsilon for normalization
    """
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        cond_dim,
        norm_eps=1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.ffn_dim = ffn_dim
        self.ffn = FeedForwardNetwork(
            self.embed_dim,
            self.ffn_dim,
        )
        self.norm = RMSNorm(self.embed_dim, eps=norm_eps)
        self.adaLN_modulation = nn.Sequential(
            # nn.SiLU(),
            ACT2FN['silu'],
            nn.Linear(cond_dim, 3 * self.embed_dim, bias=False)
        )

    def forward(self, x, c):
        shift_ffn, scale_ffn, gate_ffn = self.adaLN_modulation(c).chunk(3, dim=-1)
        x = x + gate_ffn * self.ffn(modulate(self.norm(x), shift_ffn, scale_ffn))
        return x


class FinalLayer(nn.Module):
    """
    Final layer in the diffusion head.
    
    Args:
        hidden_size (`int`): Input dimension
        output_size (`int`): Output dimension
        cond_size (`int`): Condition embedding dimension
        norm_eps (`float`, optional): Epsilon for normalization
    """
    def __init__(self, hidden_size, output_size, cond_size, norm_eps=1e-5):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.adaLN_modulation = nn.Sequential(
            # nn.SiLU(),
            ACT2FN['silu'],
            nn.Linear(cond_size, 2 * hidden_size, bias=False)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class VibeVoiceDiffusionHead(PreTrainedModel):
    """
    Diffusion head model for vibevoice.
    
    Args:
        config (`VibeVoiceDiffusionHeadConfig`): Model configuration
        latent_size (`int`, optional): Size of the latent space. If not provided, uses `config.latent_size`.
    """
    config_class = VibeVoiceDiffusionHeadConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True  
    _supports_sdpa = True  
    
    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.config = config
        self.cond_dim = config.hidden_size
        latent_size = config.latent_size
        
        self.noisy_images_proj = nn.Linear(latent_size, config.hidden_size, bias=False)
        self.cond_proj = nn.Linear(config.hidden_size, self.cond_dim, bias=False)
        self.t_embedder = TimestepEmbedder(self.cond_dim)
        
        ffn_dim = int(config.hidden_size * config.head_ffn_ratio)
        
        # Create the intermediate layers
        self.layers = nn.ModuleList([
            HeadLayer(
                embed_dim=config.hidden_size,
                ffn_dim=ffn_dim,
                cond_dim=self.cond_dim,
                norm_eps=config.rms_norm_eps
            )
            for _ in range(config.head_layers)
        ])
        
        # Final layer for output
        self.final_layer = FinalLayer(
            hidden_size=config.hidden_size, 
            output_size=latent_size,
            cond_size=self.cond_dim,
            norm_eps=config.rms_norm_eps
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the weights of the model."""
        # Initialize timestep embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for layer in self.layers:
            nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)

    def forward(
        self,
        noisy_images,
        timesteps,
        condition,
    ):
        """
        Forward pass of the prediction head.
        
        Args:
            noisy_images (`torch.Tensor`): Noisy images/latents to denoise
            timesteps (`torch.Tensor`): Timesteps for diffusion
            condition (`torch.Tensor`): Conditioning information
            
        Returns:
            `torch.Tensor`: The predicted noise/velocity
        """
        x = self.noisy_images_proj(noisy_images)
        t = self.t_embedder(timesteps)
        condition = self.cond_proj(condition)
        c = condition + t
        
        for layer in self.layers:
            x = layer(x, c)
            
        x = self.final_layer(x, c)
        return x


AutoModel.register(VibeVoiceDiffusionHeadConfig, VibeVoiceDiffusionHead)

__all__ = ["VibeVoiceDiffusionHead"]