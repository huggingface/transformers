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
        "z" : Tensor[B x D x T]
            Quantized continuous representation of input
        "codes" : Tensor[B x N x T]
            Codebook indices for each codebook
            (quantized discrete representation of input)
        "latents" : Tensor[B x N*D x T]
            Projected latents (continuous representation of input before quantization)
        "vq/commitment_loss" : Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        "vq/codebook_loss" : Tensor[1]
            Codebook loss to update the codebook
        "length" : int
            Number of samples in input audio
        "audio" : Tensor[B x 1 x length]
            Decoded audio data.
    """

    audio: torch.FloatTensor = None
    z: torch.FloatTensor = None
    codes: torch.FloatTensor = None
    latents: torch.FloatTensor = None
    commitment_loss: torch.FloatTensor = None
    codebook_loss: torch.FloatTensor = None


@dataclass
class DacEncoderOutput(ModelOutput):
    """
    Args:
        audio_scales (`torch.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
            Scaling factor for each `audio_codes` input. This is used to unscale each chunk of audio when decoding.
        z" : Tensor[B x D x T]
            Quantized continuous representation of input
        "codes" : Tensor[B x N x T]
            Codebook indices for each codebook
            (quantized discrete representation of input)
        "latents" : Tensor[B x N*D x T]
            Projected latents (continuous representation of input before quantization)
        "vq/commitment_loss" : Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        "vq/codebook_loss" : Tensor[1]
            Codebook loss to update the codebook
    """

    z: torch.FloatTensor = None
    codes: torch.FloatTensor = None
    latents: torch.FloatTensor = None
    commitment_loss: torch.FloatTensor = None
    codebook_loss: torch.FloatTensor = None


@dataclass
class DacDecoderOutput(ModelOutput):
    """
    Args:
        "audio" : Tensor[B x 1 x length]
                Decoded audio data.
    """

    audio: torch.FloatTensor = None


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class DacVectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
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

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = z_e + (z_q - z_e).detach()  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
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
        z_q = self.decode_code(indices)
        return z_q, indices


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class DacResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class DacEncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            DacResidualUnit(dim // 2, dilation=1),
            DacResidualUnit(dim // 2, dilation=3),
            DacResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class DacDecoderBlock(nn.Module):
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

    def forward(self, x):
        return self.block(x)


class DacResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
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

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)

            # Create mask to apply quantizer dropout
            mask = torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[0]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)


class DacDecoder(nn.Module):
    def __init__(
        self,
        config: DacConfig,
    ):
        super().__init__()

        input_channel = config.latent_dim
        channels = config.decoder_dim
        rates = config.decoder_rates
        d_out = 1

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
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DacEncoder(nn.Module):
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

        # Wrap black into nn.Sequential
        self.block = nn.ModuleList(self.block)
        self.enc_dim = d_model

    def forward(self, x):
        for module in self.block:
            x = module(x)
        return x


class DacPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
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
        audio_data: torch.Tensor,
        n_quantizers: int = None,
    ):
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
        """
        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(z, n_quantizers)
        return DacEncoderOutput(z, codes, latents, commitment_loss, codebook_loss)

    def decode(self, z: torch.Tensor):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input
        length : int, optional
            Number of samples in output audio, by default None

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int = None,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        length = audio_data.shape[-1]
        encoder_output = self.encode(audio_data, n_quantizers)
        x = self.decode(encoder_output.z)

        audio = x[..., :length]

        return DacOutput(audio, **encoder_output)


if __name__ == "__main__":
    model = DacModel.from_pretrained("kamilakesbi/dac_16khz")
    input_ids = [0, 4, 4, 3, 2, 4, 1, 7, 19]
    output = model(input_ids)
