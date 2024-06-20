"""PyTorch DAC model."""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm

from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput
from .configuration_dac import DacConfig


@dataclass
class DacOutput(ModelOutput):
    """
    Args:
        audio (`torch.Tensor` of shape `(batch_size, 1, input_length)`):
            Decoded audio data.
        quantized_representation (`torch.Tensor` of shape `(batch_size, dimension, time_steps)`):
            Quantized continuous representation of input.
        codebook_indices (`torch.Tensor` of shape `(batch_size, num_codebooks, time_steps)`):
            Codebook indices for each codebook (quantized discrete representation of input).
        projected_latents (`torch.Tensor` of shape `(batch_size, num_codebooks * dimension, time_steps)`):
            Projected latents (continuous representation of input before quantization).
        commitment_loss (`torch.Tensor` of shape `(1)`):
            Commitment loss to train the encoder to predict vectors closer to codebook entries.
        codebook_loss (`torch.Tensor` of shape `(1)`):
            Codebook loss to update the codebook.
    """
    audio: torch.FloatTensor = None
    quantized_representation: torch.FloatTensor = None
    codebook_indices: torch.FloatTensor = None
    projected_latents: torch.FloatTensor = None
    commitment_loss: torch.FloatTensor = None
    codebook_loss: torch.FloatTensor = None


@dataclass
class DacEncoderOutput(ModelOutput):
    """
    Args:
        quantized_representation (`torch.Tensor` of shape `(batch_size, dimension, time_steps)`, *optional*):
            Quantized continuous representation of input.
        codebook_indices (`torch.Tensor` of shape `(batch_size, num_codebooks, time_steps)`, *optional*):
            Codebook indices for each codebook (quantized discrete representation of input).
        projected_latents (`torch.Tensor` of shape `(batch_size, num_codebooks * dimension, time_steps)`, *optional*):
            Projected latents (continuous representation of input before quantization).
        commitment_loss (`torch.Tensor` of shape `(1)`, *optional*):
            Commitment loss to train the encoder to predict vectors closer to codebook entries.
        codebook_loss (`torch.Tensor` of shape `(1)`, *optional*):
            Codebook loss to update the codebook.
    """
    quantized_representation: torch.FloatTensor = None
    codebook_indices: torch.FloatTensor = None
    projected_latents: torch.FloatTensor = None
    commitment_loss: torch.FloatTensor = None
    codebook_loss: torch.FloatTensor = None


@dataclass
class DacDecoderOutput(ModelOutput):
    """
    Args:
        audio (`torch.Tensor` of shape `(batch_size, 1, input_length)`):
            Decoded audio data.
    """
    audio: torch.FloatTensor = None


def WNConv1d(*args, **kwargs):
    """1D convolution layer with Weight Normalization"""
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    """1D transposed convolution layer with Weight Normalization"""
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(input, alpha):
    shape = input.shape
    input = input.reshape(shape[0], shape[1], -1)
    input = input + (alpha + 1e-9).reciprocal() * torch.sin(alpha * input).pow(2)
    input = input.reshape(shape)
    return input


class Snake1d(nn.Module):
    """
    A 1-dimensional Snake activation function module.
    """
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, input):
        return snake(input, self.alpha)


class DacVectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo (https://github.com/karpathy/deep-vector-quantization)
    
    Additionally uses following tricks from improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, input):
        """
        Quantizes the input tensor using a fixed codebook and returns the corresponding codebook vectors.

        Args:
            input (`torch.Tensor` of shape `(batch_size, dimension, time_steps)`): 
                Input tensor.

        Returns:
            quantized_representation (`torch.Tensor`of shape `(batch_size, dimension, time_steps)`). 
                Quantized continuous representation of input, .
            commitment_loss : torch.Tensor
                Commitment loss to train encoder to predict vectors closer to codebook entries, shape `(1)`.
            codebook_loss : torch.Tensor
                Codebook loss to update the codebook, shape `(1)`.
            codebook_indices (`torch.Tensor` of shape `(batch_size, time_steps)`): 
                Codebook indices for each codebook, quantized discrete representation of input,
            projected_latents (torch.Tensor of shape `(batch_size, num_codebooks * dimension, time_steps)`): 
                Projected latents (continuous representation of input before quantization),
        """

        projected_latents = self.in_proj(input)  
        quantized_representation, codebook_indices = self.decode_latents(projected_latents)

        commitment_loss = F.mse_loss(projected_latents, quantized_representation.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(quantized_representation, projected_latents.detach(), reduction="none").mean([1, 2])

        quantized_representation = projected_latents + (quantized_representation - projected_latents).detach()  

        quantized_representation = self.out_proj(quantized_representation)

        return quantized_representation, commitment_loss, codebook_loss, codebook_indices, projected_latents

    def embed_code(self, embed_id):
        """Embeds a code using the codebook weights.
        """
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        """Decodes embedded codes by embedding them and then transposing the dimensions.
        """
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        quantized_representation = self.decode_code(indices)
        return quantized_representation, indices


class DacResidualUnit(nn.Module):
    """
    A residual unit composed of Snake1d and weight-normalized Conv1d layers with dilations.
    """
    def __init__(self, dimension: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dimension),
            WNConv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dimension),
            WNConv1d(dimension, dimension, kernel_size=1),
        )

    def forward(self, input_tensor):
        """
        Forward pass through the residual unit.

        Args:
            input_tensor (`torch.Tensor` of shape `(batch_size, channels, time_steps)`): 
                Input tensor .

        Returns:
            output_tensor (`torch.Tensor` of shape `(batch_size, channels, time_steps)`)
                Input tensor after passing through the residual unit.
        """
        output_tensor = self.block(input_tensor)
        padding = (input_tensor.shape[-1] - output_tensor.shape[-1]) // 2
        if padding > 0:
            input_tensor = input_tensor[..., padding:-padding]
        output_tensor = input_tensor + output_tensor
        return output_tensor


class DacEncoderBlock(nn.Module):
    """Encoder block used in DAC encoder."""
    def __init__(self, dimension: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            DacResidualUnit(dimension // 2, dilation=1),
            DacResidualUnit(dimension // 2, dilation=3),
            DacResidualUnit(dimension // 2, dilation=9),
            Snake1d(dimension // 2),
            WNConv1d(
                dimension // 2,
                dimension,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, input):
        return self.block(input)


class DacDecoderBlock(nn.Module):
    """Decoder block used in DAC decoder."""
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            DacResidualUnit(output_dim, dilation=1),
            DacResidualUnit(output_dim, dilation=3),
            DacResidualUnit(output_dim, dilation=9),
        )

    def forward(self, input):
        return self.block(input)


class DacResidualVectorQuantize(nn.Module):
    """
    ResidualVectorQuantize block - Introduced in SoundStream: An end2end neural audio codec (https://arxiv.org/abs/2107.03312)
    """
    def __init__(
        self,
        config: DacConfig,
    ):
        super().__init__()

        input_dim = config.latent_dim
        n_codebooks = config.n_codebooks
        codebook_size = config.codebook_size
        codebook_dim = config.codebook_dim
        quantizer_dropout = config.quantizer_dropout

        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [DacVectorQuantize(input_dim, codebook_size, codebook_dim[i]) for i in range(n_codebooks)]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, input, n_quantizers: int = None):
        """
        Quantizes the input tensor using a fixed set of codebooks and returns corresponding codebook vectors.
        Args:
            input (`torch.Tensor` of shape `(batch_size, dimension, time_steps)`):
                Input tensor to be quantized.
            n_quantizers (`int`, optional):
                Number of quantizers to use. If specified and `self.quantizer_dropout` is True,
                this argument is ignored during training, and a random number of quantizers is used.

        Returns:
            quantized_representation (`torch.Tensor` of shape `(batch_size, dimension, time_steps)`, *optional*):
                Quantized continuous representation of input.
            codebook_indices (`torch.Tensor` of shape `(batch_size, num_codebooks, time_steps)`, *optional*):
                Codebook indices for each codebook (quantized discrete representation of input).
            projected_latents (`torch.Tensor` of shape `(batch_size, num_codebooks * dimension, time_steps)`, *optional*):
                Projected latents (continuous representation of input before quantization).
            commitment_loss (`torch.Tensor` of shape `(1)`, *optional*):
                Commitment loss to train the encoder to predict vectors closer to codebook entries.
            codebook_loss (`torch.Tensor` of shape `(1)`, *optional*):
                Codebook loss to update the codebook.
        """

        quantized_representation = 0
        residual = input
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        projected_latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((input.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (input.shape[0],))
            n_dropout = int(input.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(input.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            quantized_representation_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)

            # Create mask to apply quantizer dropout
            mask = torch.full((input.shape[0],), fill_value=i, device=input.device) < n_quantizers
            quantized_representation = quantized_representation + quantized_representation_i * mask[:, None, None]
            residual = residual - quantized_representation_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            projected_latents.append(z_e_i)

        codebook_indices = torch.stack(codebook_indices, dim=1)
        projected_latents = torch.cat(projected_latents, dim=1)

        return quantized_representation, codebook_indices, projected_latents, commitment_loss, codebook_loss

    def from_codes(self, codebook_indices: torch.Tensor):
        """
        Reconstructs the continuous representation from quantized codes.

        Args:
            codebook_indices (`torch.Tensor` of shape `(batch_size, num_codebooks, time_steps)`):
                Quantized discrete representation of input.

        Returns:
            quantized_representation (`torch.Tensor`):
                Quantized continuous representation of input.
            projected_latents (`torch.Tensor`):
                List of projected latents (continuous representations of input before quantization)
                for each codebook.
            codebook_indices (`torch.Tensor`):
                Codebook indices for each codebook.
        """
        quantized_representation = 0.0
        projected_latents = []
        n_codebooks = codebook_indices.shape[1]
        for i in range(n_codebooks):
            projected_latents_i = self.quantizers[i].decode_code(codebook_indices[:, i, :])
            projected_latents.append(projected_latents_i)

            quantized_representation_i = self.quantizers[i].out_proj(projected_latents_i)
            quantized_representation = quantized_representation + quantized_representation_i
        return quantized_representation, torch.cat(projected_latents, dim=1), codebook_indices

    def from_latents(self, latents: torch.Tensor):
        """Reconstructs the quantized representation from unquantized latents.

        Args:
            latents (`torch.Tensor` of shape `(batch_size, total_latent_dimension, time_steps)`):
                Continuous representation of input after projection.

        Returns:
            quantized_representation (`torch.Tensor`, (batch_size, dimension, time_steps)):
                Quantized representation of the full-projected space.
            quantized_latents (`torch.Tensor`, (batch_size, dimension, time_steps)):
                Quantized representation of the latent space (continuous representation before quantization).
        """
        quantized_representation = 0
        quantized_latents = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[0]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            quantized_latents_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            quantized_latents.append(quantized_latents_i)
            codes.append(codes_i)

            quantized_representation_i = self.quantizers[i].out_proj(quantized_latents_i)
            quantized_representation = quantized_representation + quantized_representation_i

        return quantized_representation, torch.cat(quantized_latents, dim=1), torch.stack(codes, dim=1)


class DacDecoder(nn.Module):
    """DAC Decoder"""
    def __init__(
        self,
        config: DacConfig,
    ):
        super().__init__()

        input_channel = config.latent_dim
        channels = config.decoder_dim
        rates = config.decoder_rates

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DacDecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


class DacEncoder(nn.Module):
    """DAC Encoder"""
    def __init__(self, config: DacConfig):
        super().__init__()

        d_model = config.encoder_dim
        strides = config.encoder_rates
        d_latent = config.latent_dim
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [DacEncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        self.block = nn.ModuleList(self.block)
        self.enc_dim = d_model

    def forward(self, input):
        for module in self.block:
            input = module(input)
        return input


class DacPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    """
    config_class = DacConfig
    base_model_prefix = "dac"
    main_input_name = "input_values"

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)


class DacModel(DacPreTrainedModel):
    def __init__(self, config: DacConfig):
        super().__init__(config)
        self.config = config

        self.encoder = DacEncoder(config)
        self.decoder = DacDecoder(config)

        self.quantizer = DacResidualVectorQuantize(config)

        self.bits_per_codebook = int(math.log2(self.config.codebook_size))
        if 2**self.bits_per_codebook != self.config.codebook_size:
            raise ValueError("The codebook_size must be a power of 2.")

        # Initialize weights and apply final processing
        self.post_init()

    def encode(
        self,
        input_values: torch.Tensor,
        n_quantizers: int = None,
    ):
        """
        Encode given audio data and return quantized latent codes

        Args:
            input_values (`torch.Tensor of shape `(batch_size, 1, time_steps)`): 
                Input audio data to encode,
            num_quantizers (int, *optional*): 
                Number of quantizers to use. If None, all quantizers are used. Default is None.

        Returns:
            quantized_representation (`torch.Tensor` of shape `(batch_size, dimension, time_steps)`):
                Quantized continuous representation of input.
            codebook_indices (`torch.Tensor` of shape `(batch_size, num_codebooks, time_steps)`):
                Codebook indices for each codebook (quantized discrete representation of input),
            projected_latents (`torch.Tensor` of shape `(batch_size, num_codebooks * dimension, time_steps)`):
                Projected latents (continuous representation of input before quantization),
            commitment_loss (`torch.Tensor` of shape `(1)`):
                Commitment loss to train encoder to predict vectors closer to codebook entries.
            codebook_loss (`torch.Tensor` of shape `(1)`):
                Codebook loss to update the codebook.
        """
        quantized_representation = self.encoder(input_values)
        quantized_representation, codebook_indices, projected_latents, commitment_loss, codebook_loss = self.quantizer(quantized_representation, n_quantizers)
        return DacEncoderOutput(quantized_representation, codebook_indices, projected_latents, commitment_loss, codebook_loss)

    def decode(self, quantized_representation: torch.Tensor):
        """Decode given latent codes and return audio data

        Args:
            quantized_representation (torch.Tensor of shape `(batch_size, dimension, time_steps)`):
                Quantized continuous representation of input

        Returns:
            audio (`torch.Tensor` of shape `(batch_size, 1, input_length)`):
                Decoded audio data.
        """
        return self.decoder(quantized_representation)

    def forward(
        self,
        input_values: torch.Tensor,
        n_quantizers: int = None,
    ):
        """
        Encode given audio data and return quantized latent codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, 1, time_steps)`).
                Audio data to encode, 
            sample_rate (`int`, *optional*):
                Sample rate of audio data in Hz. If `None`, defaults to `self.sample_rate`.
            num_quantizers (`int`, *optional*):
                Number of quantizers to use. If `None`, all quantizers are used. Default is `None`.

        Returns:
            dict`: A dictionary with the following keys:

            quantized_representation (`torch.Tensor` of shape `(batch_size, dimension, time_steps)`):
                Quantized continuous representation of input, 
            codebook_indices: (`torch.Tensor` of shape `(batch_size, num_codebooks, time_steps)`):
                Codebook indices for each codebook (quantized discrete representation of input).
            projected_latents (torch.Tensor` of shape `(batch_size, num_codebooks * dimension, time_steps)`):
                Projected latents (continuous representation of input before quantization),
                shape .
            commitment_loss (`torch.Tensor`):
                Commitment loss to train encoder to predict vectors closer to codebook entries, shape `(1)`.
            codebook_loss: (`torch.Tensor`):
                Codebook loss to update the codebook, shape `(1)`.
            input_length (`int`):
                Number of samples in the input audio.
            audio (`torch.Tensor` of shape `(batch_size, 1, input_length)`):
                Decoded audio data. 
        """

        length = input_values.shape[-1]
        encoder_output = self.encode(input_values, n_quantizers)
        x = self.decode(encoder_output.z)

        audio = x[..., :length]

        return DacOutput(audio, **encoder_output)


