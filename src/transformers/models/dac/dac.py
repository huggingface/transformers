import math
from typing import List
from typing import Union


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm


import inspect
import shutil
import tempfile
import typing
from pathlib import Path

import torch
from torch import nn


class BaseModel(nn.Module):
    """This is a class that adds useful save/load functionality to a
    ``torch.nn.Module`` object. ``BaseModel`` objects can be saved
    as ``torch.package`` easily, making them super easy to port between
    machines without requiring a ton of dependencies. Files can also be
    saved as just weights, in the standard way.

    >>> class Model(ml.BaseModel):
    >>>     def __init__(self, arg1: float = 1.0):
    >>>         super().__init__()
    >>>         self.arg1 = arg1
    >>>         self.linear = nn.Linear(1, 1)
    >>>
    >>>     def forward(self, x):
    >>>         return self.linear(x)
    >>>
    >>> model1 = Model()
    >>>
    >>> with tempfile.NamedTemporaryFile(suffix=".pth") as f:
    >>>     model1.save(
    >>>         f.name,
    >>>     )
    >>>     model2 = Model.load(f.name)
    >>>     out2 = seed_and_run(model2, x)
    >>>     assert torch.allclose(out1, out2)
    >>>
    >>>     model1.save(f.name, package=True)
    >>>     model2 = Model.load(f.name)
    >>>     model2.save(f.name, package=False)
    >>>     model3 = Model.load(f.name)
    >>>     out3 = seed_and_run(model3, x)
    >>>
    >>> with tempfile.TemporaryDirectory() as d:
    >>>     model1.save_to_folder(d, {"data": 1.0})
    >>>     Model.load_from_folder(d)

    """

    EXTERN = [
        "audiotools.**",
        "tqdm",
        "__main__",
        "numpy.**",
        "julius.**",
        "torchaudio.**",
        "scipy.**",
        "einops",
    ]
    """Names of libraries that are external to the torch.package saving mechanism.
    Source code from these libraries will not be packaged into the model. This can
    be edited by the user of this class by editing ``model.EXTERN``."""
    INTERN = []
    """Names of libraries that are internal to the torch.package saving mechanism.
    Source code from these libraries will be saved alongside the model."""

    def save(
        self,
        path: str,
        metadata: dict = None,
        package: bool = True,
        intern: list = [],
        extern: list = [],
        mock: list = [],
    ):
        """Saves the model, either as a torch package, or just as
        weights, alongside some specified metadata.

        Parameters
        ----------
        path : str
            Path to save model to.
        metadata : dict, optional
            Any metadata to save alongside the model,
            by default None
        package : bool, optional
            Whether to use ``torch.package`` to save the model in
            a format that is portable, by default True
        intern : list, optional
            List of additional libraries that are internal
            to the model, used with torch.package, by default []
        extern : list, optional
            List of additional libraries that are external to
            the model, used with torch.package, by default []
        mock : list, optional
            List of libraries to mock, used with torch.package,
            by default []

        Returns
        -------
        str
            Path to saved model.
        """
        sig = inspect.signature(self.__class__)
        args = {}

        for key, val in sig.parameters.items():
            arg_val = val.default
            if arg_val is not inspect.Parameter.empty:
                args[key] = arg_val

        # Look up attibutes in self, and if any of them are in args,
        # overwrite them in args.
        for attribute in dir(self):
            if attribute in args:
                args[attribute] = getattr(self, attribute)

        metadata = {} if metadata is None else metadata
        metadata["kwargs"] = args
        if not hasattr(self, "metadata"):
            self.metadata = {}
        self.metadata.update(metadata)

        if not package:
            state_dict = {"state_dict": self.state_dict(), "metadata": metadata}
            torch.save(state_dict, path)
        else:
            self._save_package(path, intern=intern, extern=extern, mock=mock)

        return path

    @property
    def device(self):
        """Gets the device the model is on by looking at the device of
        the first parameter. May not be valid if model is split across
        multiple devices.
        """
        return list(self.parameters())[0].device

    @classmethod
    def load(
        cls,
        location: str,
        *args,
        package_name: str = None,
        strict: bool = False,
        **kwargs,
    ):
        """Load model from a path. Tries first to load as a package, and if
        that fails, tries to load as weights. The arguments to the class are
        specified inside the model weights file.

        Parameters
        ----------
        location : str
            Path to file.
        package_name : str, optional
            Name of package, by default ``cls.__name__``.
        strict : bool, optional
            Ignore unmatched keys, by default False
        kwargs : dict
            Additional keyword arguments to the model instantiation, if
            not loading from package.

        Returns
        -------
        BaseModel
            A model that inherits from BaseModel.
        """
        try:
            model = cls._load_package(location, package_name=package_name)
        except:
            model_dict = torch.load(location, "cpu")
            metadata = model_dict["metadata"]
            metadata["kwargs"].update(kwargs)

            sig = inspect.signature(cls)
            class_keys = list(sig.parameters.keys())
            for k in list(metadata["kwargs"].keys()):
                if k not in class_keys:
                    metadata["kwargs"].pop(k)

            model = cls(*args, **metadata["kwargs"])
            model.load_state_dict(model_dict["state_dict"], strict=strict)
            model.metadata = metadata

        return model

    def _save_package(self, path, intern=[], extern=[], mock=[], **kwargs):
        package_name = type(self).__name__
        resource_name = f"{type(self).__name__}.pth"

        # Below is for loading and re-saving a package.
        if hasattr(self, "importer"):
            kwargs["importer"] = (self.importer, torch.package.sys_importer)
            del self.importer

        # Why do we use a tempfile, you ask?
        # It's so we can load a packaged model and then re-save
        # it to the same location. torch.package throws an
        # error if it's loading and writing to the same
        # file (this is undocumented).
        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
            with torch.package.PackageExporter(f.name, **kwargs) as exp:
                exp.intern(self.INTERN + intern)
                exp.mock(mock)
                exp.extern(self.EXTERN + extern)
                exp.save_pickle(package_name, resource_name, self)

                if hasattr(self, "metadata"):
                    exp.save_pickle(
                        package_name, f"{package_name}.metadata", self.metadata
                    )

            shutil.copyfile(f.name, path)

        # Must reset the importer back to `self` if it existed
        # so that you can save the model again!
        if "importer" in kwargs:
            self.importer = kwargs["importer"][0]
        return path

    @classmethod
    def _load_package(cls, path, package_name=None):
        package_name = cls.__name__ if package_name is None else package_name
        resource_name = f"{package_name}.pth"

        imp = torch.package.PackageImporter(path)
        model = imp.load_pickle(package_name, resource_name, "cpu")
        try:
            model.metadata = imp.load_pickle(package_name, f"{package_name}.metadata")
        except:  # pragma: no cover
            pass
        model.importer = imp

        return model

    def save_to_folder(
        self,
        folder: typing.Union[str, Path],
        extra_data: dict = None,
        package: bool = True,
    ):
        """Dumps a model into a folder, as both a package
        and as weights, as well as anything specified in
        ``extra_data``. ``extra_data`` is a dictionary of other
        pickleable files, with the keys being the paths
        to save them in. The model is saved under a subfolder
        specified by the name of the class (e.g. ``folder/generator/[package, weights].pth``
        if the model name was ``Generator``).

        >>> with tempfile.TemporaryDirectory() as d:
        >>>     extra_data = {
        >>>         "optimizer.pth": optimizer.state_dict()
        >>>     }
        >>>     model.save_to_folder(d, extra_data)
        >>>     Model.load_from_folder(d)

        Parameters
        ----------
        folder : typing.Union[str, Path]
            _description_
        extra_data : dict, optional
            _description_, by default None

        Returns
        -------
        str
            Path to folder
        """
        extra_data = {} if extra_data is None else extra_data
        model_name = type(self).__name__.lower()
        target_base = Path(f"{folder}/{model_name}/")
        target_base.mkdir(exist_ok=True, parents=True)

        if package:
            package_path = target_base / f"package.pth"
            self.save(package_path)

        weights_path = target_base / f"weights.pth"
        self.save(weights_path, package=False)

        for path, obj in extra_data.items():
            torch.save(obj, target_base / path)

        return target_base

    @classmethod
    def load_from_folder(
        cls,
        folder: typing.Union[str, Path],
        package: bool = True,
        strict: bool = False,
        **kwargs,
    ):
        """Loads the model from a folder generated by
        :py:func:`audiotools.ml.layers.base.BaseModel.save_to_folder`.
        Like that function, this one looks for a subfolder that has
        the name of the class (e.g. ``folder/generator/[package, weights].pth`` if the
        model name was ``Generator``).

        Parameters
        ----------
        folder : typing.Union[str, Path]
            _description_
        package : bool, optional
            Whether to use ``torch.package`` to load the model,
            loading the model from ``package.pth``.
        strict : bool, optional
            Ignore unmatched keys, by default False

        Returns
        -------
        tuple
            tuple of model and extra data as saved by
            :py:func:`audiotools.ml.layers.base.BaseModel.save_to_folder`.
        """
        folder = Path(folder) / cls.__name__.lower()
        model_pth = "package.pth" if package else "weights.pth"
        model_pth = folder / model_pth

        model = cls.load(model_pth, strict=strict)
        extra_data = {}
        excluded = ["package.pth", "weights.pth"]
        files = [x for x in folder.glob("*") if x.is_file() and x.name not in excluded]
        for f in files:
            extra_data[f.name] = torch.load(f, **kwargs)

        return model, extra_data


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


class VectorQuantize(nn.Module):
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

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

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


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i])
                for i in range(n_codebooks)
            ]
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

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
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

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)




def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)




class ResidualUnit(nn.Module):
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


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
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


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
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
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
def preprocess(audio_data, sample_rate):
    if sample_rate is None:
        sample_rate = 16000
    assert sample_rate == 16000

    length = audio_data.shape[-1]
    hop_length = 512 # Make sure the hop lengths match the one of the model

    right_pad = math.ceil(length / 512) * 512 - length
    audio_data = nn.functional.pad(audio_data, (0, right_pad))

    return audio_data


class DAC(BaseModel):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        
        print(self.hop_length)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)


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
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            z, n_quantizers
        )
        return z, codes, latents, commitment_loss, codebook_loss

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
        sample_rate: int = None,
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
        audio_data = self.preprocess(audio_data, sample_rate)
        z, codes, latents, commitment_loss, codebook_loss = self.encode(
            audio_data, n_quantizers
        )
        x = self.decode(z)

        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }


if __name__ == '__main__': 

    torch.random.manual_seed(0)
    model = DAC()

    model.to('cuda:2')

    # Load audio signal file
    # signal = AudioSignal('input.wav', sample_rate=16000)
    signal = torch.rand([1, 1, 1065472])
    # Encode audio signal as one long file
    # (may run out of GPU memory on long files)
    signal = signal.to(model.device)

    print("input: ", signal.shape)

    x = preprocess(signal, 16000)

    print('output to preprocessor: ', x.shape)

    with torch.no_grad():
        z, codes, latents, _, _ = model.encode(x)

        # Decode audio signal
        y = model.decode(z)

