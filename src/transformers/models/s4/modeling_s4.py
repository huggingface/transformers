# coding=utf-8
# Copyright 2021 The HuggingFace Team The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch S4 model."""


import functools
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_einops_available,
    is_opt_einsum_available,
    is_pykeops_available,
    is_scipy_available,
    replace_return_docstrings,
)
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_s4 import S4Config


logger = logging.get_logger(__name__)

# soft dependencies
if is_pykeops_available():
    try:
        import pykeops.torch as pktorch
    except ImportError:
        logger.error(
            "S4 models are not usable since `pykeops` can't be loaded. "
            "Please try to reinstall it following the instructions here: https://www.kernel-operations.io/keops/python/installation.html"
        )

if is_opt_einsum_available():
    try:
        import opt_einsum
    except ImportError:
        logger.error(
            "S4 models are not usable since `opt_einsum` can't be loaded. "
            "Please try to reinstall it following the instructions here: https://optimized-einsum.readthedocs.io/en/stable/install.html"
        )

if is_einops_available():
    try:
        import einops
    except ImportError:
        logger.error(
            "S4 models are not usable since `einops` can't be loaded. "
            "Please try to reinstall it following the instructions here: https://einops.rocks/#installation"
        )

if is_scipy_available():
    try:
        from scipy import special as scipy_special
    except ImportError:
        logger.error(
            "S4 models are not usable since `scipy` can't be loaded. "
            "Please try to reinstall it following the instructions here: https://www.scipy.org/install.html"
        )


_CHECKPOINT_FOR_DOC = "s4"
_CONFIG_FOR_DOC = "S4Config"


class S4Embedding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: S4Config):
        super().__init__()

        self.n_token = config.vocab_size
        self.d_embed = config.d_embed

        self.cutoffs = config.cutoffs + [config.vocab_size]
        self.div_val = config.div_val
        self.d_proj = config.d_model
        self.drop = nn.Dropout(config.embedding_dropout_prob) if config.embedding_dropout_prob > 0 else None

        self.emb_scale = config.d_model ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.initializer_scale = config.initializer_scale

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()

        if config.div_val == 1:
            self.emb_layers.append(nn.Embedding(self.n_token, self.d_embed))
            if self.d_proj != self.d_embed:
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(self.d_proj, self.d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = self.d_embed // (self.div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(self.d_proj, d_emb_i)))

    def forward(self, input_ids):
        if self.div_val == 1:
            embed = self.emb_layers[0](input_ids)
            embed = self.drop(embed)
            if self.d_proj != self.d_embed:
                embed = nn.functional.linear(embed, self.emb_projs[0])
        else:
            inp_flat = input_ids.view(-1)

            # Changes
            embeddings = []
            indices = torch.zeros_like(inp_flat)  # empty should work as long as cutoffs[-1] > max token
            _total_tokens = 0

            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze(-1)  # shape (_tokens,)

                _tokens = indices_i.numel()
                if _tokens == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = self.drop(emb_i) if self.drop is not None else emb_i
                emb_i = nn.functional.linear(emb_i, self.emb_projs[i])

                # Changes
                embeddings.append(emb_i)
                indices.index_put_((indices_i,), torch.arange(_tokens, device=input_ids.device) + _total_tokens)
                _total_tokens += _tokens

            embeddings = torch.cat(embeddings, dim=0)
            emb_flat = embeddings[indices]

            embed_shape = input_ids.size() + (self.d_proj,)
            embed = emb_flat.view(embed_shape)

        embed.mul_(self.emb_scale)

        return embed


""" Cauchy kernel"""


def _broadcast_dims(*tensors):
    max_dim = max([len(tensor.shape) for tensor in tensors])
    tensors = [tensor.view((1,) * (max_dim - len(tensor.shape)) + tensor.shape) for tensor in tensors]
    return tensors


def cauchy_conj(v, z, w, num=2, denom=2):
    """Pykeops version"""
    if num == 1:
        expr_num = "z * ComplexReal(v) - Real2Complex(ComplexReal(v)*ComplexReal(w) + ComplexImag(v)*ComplexImag(w))"
    elif num == 2:
        expr_num = "z * ComplexReal(v) - Real2Complex(Sum(v * w))"
    else:
        raise NotImplementedError

    if denom == 1:
        expr_denom = "ComplexMult(z-Real2Complex(ComplexReal(w)), z-Real2Complex(ComplexReal(w))) + Real2Complex(Square(ComplexImag(w)))"
    elif denom == 2:
        expr_denom = "ComplexMult(z-w, z-Conj(w))"
    else:
        raise NotImplementedError

    cauchy_mult = pktorch.Genred(
        f"ComplexDivide({expr_num}, {expr_denom})",
        # expr_num,
        # expr_denom,
        [
            "v = Vj(2)",
            "z = Vi(2)",
            "w = Vj(2)",
        ],
        reduction_op="Sum",
        axis=1,
        dtype="float32" if v.dtype == torch.cfloat else "float64",
    )

    v, z, w = _broadcast_dims(v, z, w)
    v = torch.view_as_real(v)
    z = torch.view_as_real(z)
    w = torch.view_as_real(w)

    r = 2 * cauchy_mult(v, z, w, backend="GPU")
    return torch.view_as_complex(r)


def _conj(x):
    return torch.cat([x, x.conj()], dim=-1)


""" simple nn.Module components"""


def Activation(activation=None, dim=-1):
    if activation in [None, "id", "identity", "linear"]:
        return nn.Identity()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation in ["swish", "silu"]:
        return nn.SiLU()
    elif activation == "glu":
        return nn.GLU(dim=dim)
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))


def get_initializer(name, activation=None):
    if activation in [None, "id", "identity", "linear", "modrelu"]:
        nonlinearity = "linear"
    elif activation in ["relu", "tanh", "sigmoid"]:
        nonlinearity = activation
    elif activation in ["gelu", "swish", "silu"]:
        nonlinearity = "relu"  # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(f"get_initializer: activation {activation} not supported")

    if name == "uniform":
        initializer = functools.partial(torch.nn.init.kaiming_uniform_, nonlinearity=nonlinearity)
    elif name == "normal":
        initializer = functools.partial(torch.nn.init.kaiming_normal_, nonlinearity=nonlinearity)
    elif name == "xavier":
        initializer = torch.nn.init.xavier_normal_
    elif name == "zero":
        initializer = functools.partial(torch.nn.init.constant_, val=0)
    elif name == "one":
        initializer = functools.partial(torch.nn.init.constant_, val=1)
    else:
        raise NotImplementedError(f"get_initializer: initializer type {name} not supported")

    return initializer


class TransposedLinear(nn.Module):
    """Linear module on the second-to-last dimension"""

    def __init__(self, d_input, d_output, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(d_output, d_input))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # nn.Linear default init
        # nn.init.kaiming_uniform_(self.weight, nonlinearity='linear') # should be equivalent

        if bias:
            self.bias = nn.Parameter(torch.empty(d_output, 1))
            bound = 1 / math.sqrt(d_input)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = 0.0

    def forward(self, x):
        return opt_einsum.contract("... u l, v u -> ... v l", x, self.weight) + self.bias


def LinearActivation(
    d_input,
    d_output,
    bias=True,
    zero_bias_init=False,
    transposed=False,
    initializer=None,
    activation=None,
    activate=False,  # Apply activation as part of this module
    weight_norm=False,
    **kwargs,
):
    """Returns a linear nn.Module with control over axes order, initialization, and activation"""

    # Construct core module
    linear_cls = TransposedLinear if transposed else nn.Linear
    if activation == "glu":
        d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    # Initialize weight
    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)

    # Initialize bias
    if bias and zero_bias_init:
        nn.init.zeros_(linear.bias)

    # Weight norm
    if weight_norm:
        linear = nn.utils.weight_norm(linear)

    if activate and activation is not None:
        activation = Activation(activation, dim=-2 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear


""" Misc functional utilities"""


def krylov(L, A, b, c=None, return_power=False):
    """
    Compute the Krylov matrix (b, Ab, A^2b, ...) using the squaring trick.

    If return_power=True, return A^{L-1} as well
    """
    # TODO There is an edge case if L=1 where output doesn't get broadcasted, which might be an issue if caller is expecting broadcasting semantics... can deal with it if it arises

    x = b.unsqueeze(-1)  # (..., N, 1)
    A_ = A

    AL = None
    if return_power:
        AL = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        _L = L - 1

    done = L == 1
    # loop invariant: _L represents how many indices left to compute
    while not done:
        if return_power:
            if _L % 2 == 1:
                AL = A_ @ AL
            _L //= 2

        # Save memory on last iteration
        l = x.shape[-1]
        if L - l <= l:
            done = True
            _x = x[..., : L - l]
        else:
            _x = x

        _x = A_ @ _x
        x = torch.cat([x, _x], dim=-1)  # there might be a more efficient way of ordering axes
        if not done:
            A_ = A_ @ A_

    assert x.shape[-1] == L

    if c is not None:
        x = torch.einsum("...nl, ...n -> ...l", x, c)
    x = x.contiguous()  # WOW!!
    if return_power:
        return x, AL
    else:
        return x


def power(L, A, v=None):
    """
    Compute A^L and the scan sum_i A^i v_i

    A: (..., N, N) v: (..., N, L)
    """

    I = torch.eye(A.shape[-1]).to(A)  # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1:
            I = powers[-1] @ I
        L //= 2
        if L == 0:
            break
        l *= 2
        powers.append(powers[-1] @ powers[-1])

    if v is None:
        return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = einops.rearrange(v, "... (z l) -> ... z l", z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)


""" HiPPO utilities"""


def transition(measure, N, **measure_args):
    """
    A, B transition matrices for different measures

    measure: the type of measure legt - Legendre (translated) legs - Legendre (scaled) glagt - generalized Laguerre
    (translated) lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == "lagt":
        b = measure_args.get("beta", 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    elif measure == "tlagt":
        # beta = 1 corresponds to no tilt
        b = measure_args.get("beta", 1.0)
        A = (1.0 - b) / 2 * np.eye(N) - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    elif measure == "glagt":
        alpha = measure_args.get("alpha", 0.0)
        beta = measure_args.get("beta", 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = scipy_special.binom(alpha + np.arange(N), np.arange(N))[:, None]

        L = np.exp(0.5 * (scipy_special.gammaln(np.arange(N) + alpha + 1) - scipy_special.gammaln(np.arange(N) + 1)))
        A = (1.0 / L[:, None]) * A * L[None, :]
        B = (1.0 / L[:, None]) * B * np.exp(-0.5 * scipy_special.gammaln(1 - alpha)) * beta ** ((1 - alpha) / 2)
    # Legendre (translated)
    elif measure == "legt":
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** 0.5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.0) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # LMU: equivalent to LegT up to normalization
    elif measure == "lmu":
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1)[:, None]  # / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R
    # Legendre (scaled)
    elif measure == "legs":
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy()  # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    else:
        raise NotImplementedError

    return A, B


def rank_correction(measure, N, rank=1, dtype=torch.float):
    """Return low-rank matrix L such that A + L is normal"""

    if measure == "legs":
        assert rank >= 1
        p = torch.sqrt(0.5 + torch.arange(N, dtype=dtype)).unsqueeze(0)  # (1 N)
    elif measure == "legt":
        assert rank >= 2
        p = torch.sqrt(1 + 2 * torch.arange(N, dtype=dtype))  # (N)
        p0 = p.clone()
        p0[0::2] = 0.0
        p1 = p.clone()
        p1[1::2] = 0.0
        p = torch.stack([p0, p1], dim=0)  # (2 N)
    elif measure == "lagt":
        assert rank >= 1
        p = 0.5 ** 0.5 * torch.ones(1, N, dtype=dtype)
    else:
        raise NotImplementedError

    d = p.size(0)
    if rank > d:
        p = torch.stack([p, torch.zeros(N, dtype=dtype).repeat(rank - d, d)], dim=0)  # (rank N)
    return p


def nplr(measure, N, rank=1, dtype=torch.float):
    """
    Return w, p, q, V, B such that (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype)  # (N, N)
    B = torch.as_tensor(B, dtype=dtype)[:, 0]  # (N,)

    p = rank_correction(measure, N, rank=rank, dtype=dtype)
    Ap = A + torch.sum(p.unsqueeze(-2) * p.unsqueeze(-1), dim=-3)
    w, V = torch.linalg.eig(Ap)  # (..., N) (..., N, N)
    # V w V^{-1} = A

    # Only keep one of the conjugate pairs
    w = w[..., 0::2].contiguous()
    V = V[..., 0::2].contiguous()

    V_inv = V.conj().transpose(-1, -2)

    B = opt_einsum.contract("ij, j -> i", V_inv, B.to(V))  # V^* B
    p = opt_einsum.contract("ij, ...j -> ...i", V_inv, p.to(V))  # V^* p

    return w, p, p, B, V


""" Final S4 Module"""


class OptimModule(nn.Module):
    """Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters"""

    def register(self, name, tensor, trainable=0, lr=None, wd=None, repeat=1):
        """Utility method: register a tensor as a buffer or trainable parameter"""

        if trainable == 0:
            self.register_buffer(name, tensor)
        elif trainable == 1:
            self.register_parameter(name, nn.Parameter(tensor))
        elif trainable == 2:
            tensor = tensor.repeat(repeat, *(1,) * len(tensor.shape))
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            raise NotImplementedError

        optim = {}
        if trainable and lr is not None:
            optim["lr"] = lr
            # setattr(getattr(self, name), '_lr', lr)
        if trainable and wd is not None:
            optim["weight_decay"] = wd
            # setattr(getattr(self, name), '_wd', wd)
        if len(optim) > 0:
            setattr(getattr(self, name), "_optim", optim)


class SSKernelNPLR(OptimModule):
    """
    Stores a representation of and computes the SSKernel function K_L(A^dt, B^dt, C) corresponding to a discretized
    state space, where A is Normal + Low Rank (NPLR)

    The class name stands for 'State-Space SSKernel for Normal Plus Low-Rank'. The parameters of this function are as
    follows.

    A: (... N N) the state matrix B: (... N) input matrix C: (... N) output matrix dt: (...) timescales /
    discretization step size p, q: (... P N) low-rank correction to A, such that Ap=A+pq^T is a normal matrix

    The forward pass of this Module returns: (... L) that represents represents FFT SSKernel_L(A^dt, B^dt, C)

    """

    @torch.no_grad()
    def _process_C(self, L, double_length=False):
        C = torch.view_as_complex(self.C)
        self._setup(setup_C=False)
        dA = self.dA
        dA_L = power(L, dA)
        # I = torch.eye(dA.size(-1)).to(dA)
        N = C.size(-1)
        # Multiply C by I - dA_L
        C_ = C[..., 0, :]
        C_ = torch.cat([C_, C_.conj()], dim=-1)
        prod = opt_einsum.contract("... m n, ... n -> ... m", dA_L.conj().transpose(-1, -2), C_)
        if double_length:  # Multiply by I + dA_L instead
            C_ = C_ + prod
        else:
            C_ = C_ - prod
        C_ = C_[..., :N]
        self.C[..., 0, :, :].copy_(torch.view_as_real(C_))

    def _nodes(self, L, dtype, device):
        # Cache FFT nodes and their "unprocessed" them with the bilinear transform
        # nodes = torch.tensor(np.exp(-2j * np.pi / (L)), dtype=torch.cfloat, device=Ap.device) # \omega_{2L}
        nodes = torch.tensor(np.exp(-2j * np.pi / (L)), dtype=dtype, device=device)  # \omega_{2L}
        nodes = nodes ** torch.arange(0, L // 2 + 1, device=device)
        z = 2 * (1 - nodes) / (1 + nodes)
        return nodes, z

    def __init__(
        self,
        L,
        w,
        p,
        q,
        B,
        C,
        log_dt,
        trainable=None,
        lr=None,
        setup_C=False,
        keops=False,
    ):
        """
        Optim arguments into a representation. This occurs after init so that these operations can occur after moving
        model to device

        L: Maximum length; this module computes SSKernel function of length L A: (..., N, N) represented by diag(w) -
        pq^* B: (..., N) C: (..., N) dt: (...) p: (..., N) low-rank correction to A q: (..., N)
        """

        super().__init__()
        self.keops = keops

        # Rank of low-rank correction
        assert p.shape[-2] == q.shape[-2]
        self.rank = p.shape[-2]
        self.L = L

        # Augment B and C with low rank correction
        B = B.unsqueeze(-2)  # (..., 1, N)
        C = C.unsqueeze(-2)  # (..., 1, N)
        if len(B.shape) > len(p.shape):
            p = p.repeat(B.shape[:-2] + (1, 1))
        B = torch.cat([B, p], dim=-2)
        if len(C.shape) > len(q.shape):
            q = q.repeat(C.shape[:-2] + (1, 1))
        C = torch.cat([C, q], dim=-2)

        if L is not None:
            nodes, z = self._nodes(L, dtype=w.dtype, device=w.device)
            self.register_buffer("nodes", torch.view_as_real(nodes))
            self.register_buffer("z", torch.view_as_real(z))

        # Register parameters
        if trainable is None:
            trainable = {"A": 0, "B": 0, "C": 0, "dt": 0}
        if lr is None:
            lr = {"A": None, "B": None, "C": None, "dt": None}
        repeat = C.size(0)
        self.register("log_dt", log_dt, trainable["dt"], lr["dt"], 0.0)
        self.register("w", torch.view_as_real(w), trainable["A"], lr["B"], 0.0, repeat=repeat)
        self.register("B", torch.view_as_real(B), trainable["B"], lr["B"], 0.0, repeat=repeat)
        self.register("C", torch.view_as_real(C), trainable["C"], lr["C"])

        if setup_C:
            self._process_C(L)

    def forward(self, state=None, rate=1.0, L=None):
        """
        state: (..., s, N) extra tensor that augments B rate: sampling rate factor
        """
        # if L is not None: raise NotImplementedError

        # TODO: handle potential length doubling logic so that max_len doesn't need to be passed in
        while rate == 1.0 and L > self.L:
            logger.info(f"s4: Doubling length from L = {self.L} to {2*self.L}")
            self.double_length()

        if L is None:
            L = self.L
        if rate == 1.0:
            L = self.L
        else:
            rate = self.L / L
        dt = torch.exp(self.log_dt) * rate
        B = torch.view_as_complex(self.B)
        C = torch.view_as_complex(self.C)
        w = torch.view_as_complex(self.w)  # (..., N)
        # z = torch.view_as_complex(self.z) # (..., L)

        # TODO adjust based on rate times normal max length
        if L == self.L:
            nodes = torch.view_as_complex(self.nodes)
            z = torch.view_as_complex(self.z)  # (..., L)
        else:
            nodes, z = self._nodes(L, dtype=w.dtype, device=w.device)

        # Augment B
        if state is not None:  # TODO have not updated
            # Have to "unbilinear" the state to put it into the same "type" as B
            # Compute (I + dt/2 A) @ state
            s = state.transpose(0, 1)  # (H B N)
            p = B[..., 1:, :]  # (... r N)
            q = C[..., 1:, :]  # (... r N)

            # Calculate opt_einsum.contract('... s n, ... r n, ... r m -> ... s m', sV, qV.conj(), pV), but take care of conjugate symmetry
            sA = s * w.unsqueeze(-2) - (2 + 0j) * (s @ q.conj().transpose(-1, -2)).real @ p
            s = s / dt.unsqueeze(-1).unsqueeze(-1) + sA / 2

            B = torch.cat([s, B], dim=-2)  # (..., 2+s, N)

        # Incorporate dt into A
        w = w * dt.unsqueeze(-1)  # (... N)

        # Incorporate B and C batch dimensions
        v = B.unsqueeze(-3) * C.unsqueeze(-2).conj()  # (..., 2, 2, N)
        w = w[..., None, None, :]  # (..., 1, 1, N)
        z = z[..., None, None, :]  # (..., 1, 1, L)

        # Calculate resolvent at nodes
        r = cauchy_conj(v, z, w)
        r = r * dt[..., None, None, None]  # (..., 1+r, 1+r, L)

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[..., :-1, :-1, :] - r[..., :-1, -1:, :] * r[..., -1:, :-1, :] / (1 + r[..., -1:, -1:, :])
        elif self.rank == 2:
            r00 = r[..., : -self.rank, : -self.rank, :]
            r01 = r[..., : -self.rank, -self.rank :, :]
            r10 = r[..., -self.rank :, : -self.rank, :]
            r11 = r[..., -self.rank :, -self.rank :, :]
            det = (1 + r11[..., :1, :1, :]) * (1 + r11[..., 1:, 1:, :]) - r11[..., :1, 1:, :] * r11[..., 1:, :1, :]
            s = (
                r01[..., :, :1, :] * (1 + r11[..., 1:, 1:, :]) * r10[..., :1, :, :]
                + r01[..., :, 1:, :] * (1 + r11[..., :1, :1, :]) * r10[..., 1:, :, :]
                - r01[..., :, :1, :] * (r11[..., :1, 1:, :]) * r10[..., 1:, :, :]
                - r01[..., :, 1:, :] * (r11[..., 1:, :1, :]) * r10[..., :1, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[..., : -self.rank, : -self.rank, :]
            r01 = r[..., : -self.rank, -self.rank :, :]
            r10 = r[..., -self.rank :, : -self.rank, :]
            r11 = r[..., -self.rank :, -self.rank :, :]
            r11 = einops.rearrange(r11, "... a b n -> ... n a b")
            r11 = torch.linalg.inv(torch.eye(self.rank, device=r.device) + r11)
            r11 = einops.rearrange(r11, "... n a b -> ... a b n")
            k_f = r00 - torch.einsum("... i j n, ... j k n, ... k l n -> ... i l n", r01, r11, r10)

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + nodes)

        k = torch.fft.irfft(k_f)  # (..., 1, 1+s, L)
        if state is not None:
            k_state = k[..., 0, :-1, :]  # (..., s, L)
            k_state = k_state.transpose(0, 1)
            k_B = k[..., 0, -1, :]  # (..., L)
            return k_B.to(torch.float), k_state.to(torch.float)
        else:
            return k.squeeze(-2).squeeze(-2).to(torch.float)

    @torch.no_grad()
    def double_length(self):
        self._process_C(self.L, double_length=True)

        self.L *= 2
        dtype = torch.view_as_complex(self.w).dtype
        nodes, z = self._nodes(self.L, dtype=dtype, device=self.w.device)
        self.register_buffer("nodes", torch.view_as_real(nodes))
        self.register_buffer("z", torch.view_as_real(z))

    @torch.no_grad()
    def _check(self):
        """Check if A, B, C parameters and vanilla SSKernel construction can be recovered"""

        self._setup(setup_C=True)

        K = krylov(self.L, self.dA, self.dB, self.dC.conj())

        diff = K - self.forward()
        print("checking SSKernel construction", torch.sum(diff ** 2))

    def _setup(self, setup_C=True):
        w = _conj(torch.view_as_complex(self.w))
        B = _conj(torch.view_as_complex(self.B))
        C = _conj(torch.view_as_complex(self.C))
        C = C.conj()
        p = B[..., -1, :]
        q = C[..., -1, :]
        B = B[..., 0, :]
        C = C[..., 0, :]
        dt = torch.exp(self.log_dt)
        d = (2.0 / dt.unsqueeze(-1) - w).reciprocal()  # (H, N)
        r = (1 + opt_einsum.contract("... n, ... n, ... n -> ...", q, d, p)).reciprocal()
        # A_f = torch.diag_embed(2./dt[:, None] + w) - opt_einsum.contract('... n, ... m -> ... n m', p, q)
        # A_b = torch.diag_embed(d) - opt_einsum.contract('... p, ... p, ..., ... q, ... q -> ... p q', d, p, r, q, d)
        # dA = A_b @ A_f

        self.step_params = {
            "d": d,
            "r": r.unsqueeze(-1) * d * q,
            # 'r': r,
            "p": p,
            "q": q,
            "B": B,
            "d1": 2.0 / dt.unsqueeze(-1) + w,
        }
        N = d.size(-1)
        H = dt.size(-1)

        state = torch.eye(N, dtype=w.dtype, device=w.device).unsqueeze(-2)
        u = w.new_zeros(H)
        dA = self.step_state_linear(u, state)
        dA = einops.rearrange(dA, "n h m -> h m n")
        self.dA = dA
        u = w.new_ones(H)
        state = w.new_zeros(N // 2)
        dB = self.step_state_linear(u, state)
        dB = _conj(dB)
        self.dB = dB

        if setup_C:
            dA_L = power(self.L, dA)
            I = torch.eye(dA.size(-1)).to(dA)
            dC = torch.linalg.solve(I - dA_L.transpose(-1, -2).conj(), C.conj().unsqueeze(-1)).squeeze(-1)
            self.dC = dC

    def step_state_linear(self, u=None, state=None):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step. Unfortunately, as currently
        implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel
        implementation would be much faster
        """
        N = self.step_params["d"].size(-1)
        H = self.log_dt.size(-1)

        if u is None:
            u = torch.zeros(H, dtype=torch.float, device=self.log_dt.device)
        if state is None:
            state = torch.zeros(H, N, dtype=torch.cfloat, device=self.log_dt.device)

        conj = state.size(-1) != N
        step_params = self.step_params.copy()
        if conj:
            assert state.size(-1) == N // 2
            step_params = {k: v[..., : N // 2] for k, v in step_params.items()}
        d1 = step_params["d1"]  # (H N)
        p = step_params["p"]  # (H N)
        q = step_params["q"]  # (H N)
        B = step_params["B"]  # (H N)
        r = step_params["r"]
        d = step_params["d"]  # (H N)
        # dC = self.step_params['dC'] # (H N)
        state = state.to(d1)

        if conj:
            new_state = 2 * p * torch.sum(q * state, dim=-1, keepdim=True).real  # conjugated version
        else:
            new_state = opt_einsum.contract("... n, ... m, ... m -> ... n", p, q, state)  # (B H N)
        new_state = d1 * state - new_state
        new_state = new_state + 2.0 * B * u.unsqueeze(-1)  # (B H N)
        if conj:
            A_ = 2 * p * torch.sum(r * new_state, dim=-1, keepdim=True).real  # conj version
        else:
            A_ = opt_einsum.contract("... p, ... q, ... q -> ... p", p, r, new_state)  # (B H N)
        new_state = d * (new_state - A_)

        return new_state

    def step_state(self, u, state):
        state = state.to(self.dA)
        conj = state.size(-1) != self.dA.size(-1)
        if conj:
            state = _conj(state)
        next_state = opt_einsum.contract("h m n, b h n -> b h m", self.dA, state) + opt_einsum.contract(
            "h n, b h -> b h n", self.dB, u
        )
        if conj:
            next_state = next_state[..., : state.size(-1) // 2]
        return next_state

    def step(self, u, state, linear=False):

        N = self.step_params["d"].size(-1)
        conj = state.size(-1) != N

        if linear:
            new_state = self.step_state_linear(u, state)
        else:
            new_state = self.step_state(u, state)

        if conj:
            assert state.size(-1) == N // 2
            # dC = self.dC[..., 0::2].conj()
            dC = self.dC[..., : N // 2].conj()
            out = 2 * torch.sum(dC * new_state, dim=-1).real  # conj version
        else:
            out = opt_einsum.contract("... n, ... n -> ...", self.dC.conj(), new_state)
        return out.to(torch.float), new_state


class HippoSSKernel(nn.Module):
    """Wrapper around SSKernelNPLR that generates A, B, C, dt according to HiPPO arguments."""

    def __init__(
        self,
        N,
        H,
        L=None,
        measure="legs",
        rank=1,
        dt_min=0.001,
        dt_max=0.1,
        trainable=None,
        lr=None,
        length_correction=False,
        precision=1,
        cache=False,
        resample=False,  # if given inputs of different lengths, adjust the sampling rate
        keops=False,
    ):
        super().__init__()
        self.N = N
        self.H = H
        L = L or 1
        self.precision = precision
        dtype = torch.double if self.precision == 2 else torch.float
        self.rate = None if resample else 1.0

        # Set default trainable and lr parameters
        self.trainable = {
            "A": 1,
            "B": 2,
            "C": 1,
            "dt": 1,
        }
        if trainable is not None:
            self.trainable.update(trainable)
        self.lr = {
            "A": 1e-3,
            "B": 1e-3,
            "C": None,
            "dt": 1e-3,
        }

        if lr is not None:
            self.lr.update(lr)

        # Generate dt
        self.log_dt = torch.rand(self.H, dtype=dtype) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)

        # Compute the preprocessed representation
        # Generate low rank correction p for the measure
        w, p, q, B, _ = nplr(measure, N, rank, dtype=dtype)
        cdtype = torch.cfloat if dtype == torch.float else torch.cdouble
        C = torch.randn(self.H, self.N // 2, dtype=cdtype)

        self.krylov = SSKernelNPLR(
            L,
            w,
            p,
            q,
            B,
            C,
            self.log_dt,
            trainable=self.trainable,
            lr=self.lr,
            setup_C=length_correction,
            keops=keops,
        )

        # Cached tensors
        self.K = None
        self.cache = cache

    def forward(self, state=None, L=None):
        """
        state: (B, H, N)
        """

        if state is not None:
            k, k_state = self.krylov(state=state, rate=self.rate, L=L)  # (B, H, L) (B, H, N)
            return k, k_state
        else:
            # Calculate K if needed
            if not self.training and self.K is not None and self.K.size(-1) == L:
                k = self.K
            else:
                k = self.krylov(rate=self.rate, L=L).to(torch.float)

            # Store K if needed
            if self.cache and not self.training:
                self.K = k
            else:  # If training, parameter will change after backprop so make sure to recompute on next pass
                self.K = None
            return k

    @torch.no_grad()
    def next_state(self, state, u):
        """
        state: (..., N) u: (..., L)

        Returns: (..., N)
        """

        self.krylov._setup()
        dA, dB = self.krylov.dA, self.krylov.dB

        conj = state.size(-1) != dA.size(-1)
        if conj:
            state = _conj(state)

        v = dB.unsqueeze(-1) * u.flip(-1).unsqueeze(-2)  # (..., N, L)
        AL, v = power(u.size(-1), dA, v)
        next_state = opt_einsum.contract("... m n, ... n -> ... m", AL, state)
        next_state = next_state + v

        if conj:
            next_state = next_state[..., : next_state.size(-1) // 2]
        return next_state

    def step(self, u, state):
        return self.krylov.step(u, state)

    def double_length(self):
        self.krylov.double_length()


class S4(nn.Module):
    def __init__(
        self,
        H,
        l_max=None,
        # Arguments for SSM Kernel
        d_state=64,
        measure="legs",
        dt_min=0.001,
        dt_max=0.1,
        rank=1,
        trainable=None,
        lr=None,
        length_correction=False,
        stride=1,
        weight_decay=0.0,  # weight decay on the SS Kernel
        precision=1,
        cache=False,  # Cache the SS Kernel during evaluation
        # Arguments for FF
        activation="gelu",  # activation in between SS and FF
        postact=None,  # activation after FF
        weight_norm=False,  # weight normalization on FF
        initializer=None,  # initializer on FF
        input_linear=False,
        hyper_act=None,
        dropout=0.0,
        transposed=True,  # axis ordering (B, L, D) or (B, D, L)
        resample=False,
        use_state=False,
        verbose=False,
        keops=False,
    ):
        """
        d_state: the dimension of the state, also denoted by N l_max: the maximum sequence length, also denoted by L if
        this is not known at model creation, or inconvenient to pass in, set l_max=None and length_correction=True
        dropout: standard dropout argument transposed: choose backbone axis ordering of (B, L, D) or (B, D, L) [B=batch
        size, L=sequence length, D=feature dimension]

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        if verbose:
            logger.info(f"Constructing s4 (H, N, L) = ({H}, {d_state}, {l_max})")

        self.h = H
        self.n = d_state if d_state > 0 else H
        self.stride = stride
        if l_max is not None and stride > 1:
            assert l_max % stride == 0
            l_max = l_max // self.stride
        self.cache = cache
        self.weight_decay = weight_decay
        self.transposed = transposed
        self.resample = resample

        self.D = nn.Parameter(torch.randn(self.h))

        # Optional (position-wise) input transform
        if input_linear:
            self.input_linear = LinearActivation(
                self.h,
                self.h,
                transposed=self.transposed,
                initializer=initializer,
                activation=postact,
                activate=True,
                weight_norm=weight_norm,
            )
        else:
            self.input_linear = nn.Identity()

        # SSM Kernel
        self.kernel = HippoSSKernel(
            self.n,
            self.h,
            l_max,
            dt_min=dt_min,
            dt_max=dt_max,
            measure=measure,
            rank=rank,
            trainable=trainable,
            lr=lr,
            length_correction=length_correction,
            precision=precision,
            cache=cache,
            resample=resample,
            keops=keops,
        )
        self.K = None  # Cache the computed convolution filter if possible (during evaluation)

        # optional multiplicative modulation
        self.hyper = hyper_act is not None
        if self.hyper:
            self.hyper_linear = LinearActivation(
                self.h,
                self.h,
                transposed=True,
                initializer=initializer,
                activation=hyper_act,
                activate=True,
                weight_norm=weight_norm,
            )

        self.activation = Activation(activation)
        dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            self.h,
            self.h,
            transposed=self.transposed,
            initializer=initializer,
            activation=postact,
            activate=True,
            weight_norm=weight_norm,
        )

        if use_state:
            self._initial_state = nn.Parameter(torch.zeros(self.h, self.n))

    def forward(self, u, state=None, cache=None, **kwargs):  # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H) state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        u = self.input_linear(u)
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SS Kernel
        if state is not None:
            assert self.stride == 1, "Striding not supported with states"
            k, k_state = self.kernel(state=state, L=L)
        else:
            k = self.kernel(L=L)

        # Stride the filter if needed
        if self.stride > 1:
            k = k[..., : L // self.stride]  # (H, L/S)
            k = nn.functional.pad(k.unsqueeze(-1), (0, self.stride - 1))  # (H, L/S, S)
            k = einops.rearrange(k, "... h s -> ... (h s)")  # (H, L)
        else:
            k = k[..., :L]

        # Convolution
        k_f = torch.fft.rfft(k, n=2 * L)  # (H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        y_f = k_f * u_f
        y = torch.fft.irfft(y_f, n=2 * L)[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        # Compute state update
        if state is not None:
            y = y + k_state[..., :L]
            next_state = self.kernel.next_state(state, u)
        else:
            next_state = None

        # Optional hyper-network multiplication
        if self.hyper:
            hyper = self.hyper_linear(u)
            y = hyper * y

        y = self.dropout(self.activation(y))

        if not self.transposed:
            y = y.transpose(-1, -2)

        y = self.output_linear(y)

        return y, next_state

    def step(self, u, state):
        """
        Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H) state: (B H N) Returns: output (B H), state (B H N)
        """
        assert not self.training
        y, next_state = self.kernel.step(u, state)
        y = y + u * self.D
        y = self.output_linear(self.activation(y).unsqueeze(-1)).squeeze(-1)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        return self._initial_state.repeat(*batch_shape, 1, 1)

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: einops.rearrange("... h n -> ... (h n)", state)


class TransposedLN(nn.Module):
    """
    LayerNorm module over second-to-last dimension

    This is slow and a dedicated CUDA/Triton implementation shuld provide substantial end-to-end speedup
    """

    def __init__(self, d, scalar=True):
        super().__init__()
        self.scalar = scalar
        if self.scalar:
            self.m = nn.Parameter(torch.zeros(1))
            self.s = nn.Parameter(torch.ones(1))
        else:
            self.ln = nn.LayerNorm(d)

    def forward(self, x):
        if self.scalar:
            s, m = torch.std_mean(x, dim=-2, unbiased=False, keepdim=True)
            y = (self.s / s) * (x - m + self.m)
        else:
            y = self.ln(x.transpose(-1, -2)).transpose(-1, -2)
        return y


class Normalization(nn.Module):
    def __init__(self, d, transposed=False, _name_="layer", **kwargs):  # Length dimension is -1 or -2
        super().__init__()
        self.transposed = transposed

        if _name_ == "layer":
            self.channel = True  # Normalize over channel dimension
            if self.transposed:
                self.norm = TransposedLN(d, **kwargs)
            else:
                self.norm = nn.LayerNorm(d, **kwargs)
        elif _name_ == "instance":
            self.channel = False
            norm_args = {"affine": False, "track_running_stats": False}
            norm_args.update(kwargs)
            self.norm = nn.InstanceNorm1d(d, **norm_args)  # (True, True) performs very poorly
        elif _name_ == "batch":
            self.channel = False
            norm_args = {"affine": True, "track_running_stats": True}
            norm_args.update(kwargs)
            self.norm = nn.BatchNorm1d(d, **norm_args)
        elif _name_ == "none":
            self.channel = True
            self.norm = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x):
        # The cases of LayerNorm / no normalization are automatically handled in all cases
        # Instance/Batch Norm work automatically with transposed axes
        if self.channel or self.transposed:
            return self.norm(x)
        else:
            x = x.transpose(-1, -2)
            x = self.norm(x)
            x = x.transpose(-1, -2)
            return x


class FF(nn.Module):
    def __init__(
        self, d_input, expand=2, d_output=None, transposed=False, activation="gelu", initializer=None, dropout=0.0
    ):
        super().__init__()
        self.d_output = d_input if d_output is None else d_output
        self.transposed = transposed
        d_inner = expand * d_input

        linear1 = LinearActivation(
            d_input,
            d_inner,
            transposed=transposed,
            activation=activation,
            initializer=initializer,
            activate=True,
        )
        dropout_cls = nn.Dropout2d if self.transposed else nn.Dropout
        drop = dropout_cls(dropout) if dropout > 0.0 else nn.Identity()

        linear2 = LinearActivation(
            d_inner,
            self.d_output,
            transposed=transposed,
            activation=None,
            initializer=initializer,
            activate=False,
        )

        self.ff = nn.Sequential(
            linear1,
            drop,
            linear2,
        )

    def forward(self, x, state=None):
        return self.ff(x), None

    def step(self, x, state):
        # x: [batch, d_input]
        if self.transposed:
            # expects: [batch, d_input, seq_len]
            return self.ff(x.unsqueeze(-1)).squeeze(-1), state
        else:
            return self.ff(x), state


def downsample(x, pool=1, expand=1, transposed=False):
    if x is None:
        return None
    if pool > 1:
        if transposed:
            x = x[..., 0::pool]
        else:
            x = x[..., 0::pool, :]

    if expand > 1:
        if transposed:
            x = einops.repeat(x, "... d l -> ... (d e) l", e=expand)
        else:
            x = einops.repeat(x, "... d -> ... (d e)", e=expand)

    return x


class DownSample(nn.Module):
    def __init__(self, d_input, pool=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.pool = pool
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        return downsample(x, self.pool, self.expand, self.transposed)

    def step(self, x, state):
        if self.pool > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        return self.d_input * self.expand


class Residual(nn.Module):
    """Residual connection with constant affine weights. Can simulate standard residual, no residual, and "constant gates"."""

    def __init__(self, i_layer, d_input, d_model, alpha=1.0, beta=1.0):
        # print("ConstantResidual extra kwargs", kwargs)
        super().__init__()
        assert (d_input == d_model) or alpha == 0.0
        self.i_layer = i_layer
        self.d_input = d_input
        self.d_model = d_model
        self.alpha = alpha
        self.beta = beta

    @property
    def d_output(self):
        return self.d_model

    def forward(self, x, y, transposed):  # TODO documentation of transposed
        y = self.beta * y if self.beta != 1.0 else y
        return self.alpha * x + y if self.alpha else y


class SequenceResidualBlock(nn.Module):
    def __init__(
        self,
        d_input,
        i_layer=None,  # Only needs to be passed into certain residuals like Decay
        prenorm=True,
        dropout=0.0,
        layer=None,  # Config for black box module
        residual=False,  # Config for residual function
        norm=None,  # Config for normalization layer
        pool_size=1,
        pool_expand=1,
    ):
        super().__init__()

        self.i_layer = i_layer
        self.d_input = d_input
        self.layer = layer
        self.prenorm = prenorm

        # Residual
        # d_residual is the output dimension after residual
        if residual is False:
            self.residual = None
            self.d_residual = self.layer.d_output
        else:
            self.residual = Residual(i_layer, d_input, self.layer.d_output)
            self.d_residual = self.residual.d_output

        # Normalization
        d_norm = d_input if self.prenorm else self.d_residual
        # We don't use config to directly instantiate since Normalization has some special cases
        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            self.norm = Normalization(d_norm, transposed=self.transposed, _name_=norm)
        else:
            self.norm = Normalization(d_norm, transposed=self.transposed, **norm)

        # Pool
        self.pool = DownSample(self.d_residual, pool_size, pool_expand, self.transposed)

        # Dropout
        drop_cls = nn.Dropout2d if self.transposed else nn.Dropout
        self.drop = drop_cls(dropout) if dropout > 0.0 else nn.Identity()

    @property
    def transposed(self):
        return getattr(self.layer, "transposed", False)

    @property
    def d_output(self):
        return self.pool.d_output if self.pool is not None else self.d_residual

    def forward(self, x, *args, state=None, **kwargs):
        y = x

        # Pre-norm
        if self.norm is not None and self.prenorm:
            y = self.norm(y)

        # Black box module
        y, state = self.layer(y, *args, state=state, **kwargs)

        # Residual
        if self.residual is not None:
            x = self.residual(x, self.drop(y), self.transposed)

        # Post-norm
        if self.norm is not None and not self.prenorm:
            x = self.norm(x)

        # Pool
        # x = pool.downpool(x, self.pool, self.expand, self.transposed)
        if self.pool is not None:
            x = self.pool(x)

        return x, state

    def step(self, x, state, *args, **kwargs):  # TODO needs fix for transpose logic
        y = x

        # Pre-norm
        if self.norm is not None and self.prenorm:
            if self.transposed:
                y = y.unsqueeze(-1)
            y = self.norm(y)  # TODO transpose seems wrong
            if self.transposed:
                y = y.squeeze(-1)

        # Black box module
        y, state = self.layer.step(y, state, *args, **kwargs)

        # Residual
        if self.residual is not None:
            x = self.residual(x, y, transposed=False)  # TODO this would not work with concat

        # Post-norm
        if self.norm is not None and not self.prenorm:
            if self.transposed:
                y = y.unsqueeze(-1)
            x = self.norm(x)  # .step(x)
            if self.transposed:
                y = y.squeeze(-1)

        # Pool
        if self.pool is not None:
            x = self.pool(x)

        return x, state


class OptionalParameterList(nn.ParameterList):
    def extra_repr(self):
        child_lines = []
        for k, p in self._parameters.items():
            if p is not None:
                size_str = "x".join(str(size) for size in p.size())
                device_str = "" if not p.is_cuda else " (GPU {})".format(p.get_device())
                parastr = "Parameter containing: [{} of size {}{}]".format(torch.typename(p), size_str, device_str)
                child_lines.append("  (" + str(k) + "): " + parastr)
        tmpstr = "\n".join(child_lines)
        return tmpstr


class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(
        self,
        n_token,
        d_embed,
        d_proj,
        cutoffs,
        div_val=1,
        tie_projs=None,
        out_layers_weights=None,
        out_projs=None,
        keep_order=False,
        dropout=0.0,
    ):
        super().__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = list(cutoffs) + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        # [21-09-15 AG]: bake the first False into the definition, just as [0] is built into the cutoffs
        if tie_projs is None:
            tie_projs = []
        elif isinstance(tie_projs, bool):
            tie_projs = [tie_projs] * len(cutoffs)
        else:
            tie_projs = list(tie_projs)
        tie_projs = [False] + tie_projs
        self.tie_projs = tie_projs

        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        if not out_layers_weights:
            self.out_layers_weights = nn.ParameterList()
        else:
            self.out_layers_weights = out_layers_weights

        self.out_layers_biases = nn.ParameterList()

        self.shared_out_projs = out_projs
        self.out_projs = OptionalParameterList()

        self.dropout = dropout
        self.drop = nn.Dropout(dropout)

        if div_val == 1:
            if d_proj != d_embed:
                for i in range(len(self.cutoffs)):
                    if tie_projs[i]:
                        self.out_projs.append(None)
                    else:
                        self.out_projs.append(nn.Parameter(torch.zeros(d_proj, d_embed)))
            else:
                # self.out_projs = [None] * len(self.cutoffs)
                self.out_projs.append(None)

            self.out_layers_biases.append(nn.Parameter(torch.zeros(n_token)))

            if not out_layers_weights:
                self.out_layers_weights.append(nn.Parameter(torch.zeros(n_token, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val ** i)

                if tie_projs[i]:
                    self.out_projs.append(None)
                else:
                    self.out_projs.append(nn.Parameter(torch.zeros(d_proj, d_emb_i)))

                self.out_layers_biases.append(nn.Parameter(torch.zeros(r_idx - l_idx)))
                if not out_layers_weights:
                    self.out_layers_weights.append(nn.Parameter(torch.zeros(r_idx - l_idx, d_emb_i)))

        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = nn.functional.linear(hidden, weight, bias=bias)
        else:
            if self.dropout > 0.0:
                logit = hidden @ proj
                logit = self.drop(logit)
                logit = logit @ weight.t()
            else:
                logit = torch.einsum("bd,de,ev->bv", (hidden, proj, weight.t()))
            if bias is not None:
                logit = logit + bias
        return logit

    def get_out_proj(self, i):
        if self.tie_projs[i]:
            if len(self.shared_out_projs) == 0:
                return None
            elif len(self.shared_out_projs) == 1:
                return self.shared_out_projs[0]
            else:
                return self.shared_out_projs[i]
        else:
            return self.out_projs[i]

    def forward(self, hidden, target=None, keep_order=False, key_padding_mask=None, *args, **kwargs):
        # [21-09-15 AG]: TODO may need to handle key_padding_mask
        """
        hidden :: [len*bsz x d_proj] target :: [len*bsz]
        """

        hidden = hidden.reshape(-1, hidden.size(-1))
        if target is not None:
            target = target.reshape(-1)
            if hidden.size(0) != target.size(0):
                print(hidden.shape, target.shape)
                raise RuntimeError("Input and target should have the same size " "in the batch dimension.")
        nll = None
        if self.n_clusters == 0:
            logit = self._compute_logit(
                hidden, self.out_layers_weights[0], self.out_layers_biases[0], self.get_out_proj(0)
            )
            if target is not None:
                # Shift so that tokens < n predict n
                shift_logits = logit[..., :-1, :].contiguous()
                target = target[..., 1:].contiguous()
                nll = -nn.functional.log_softmax(shift_logits, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
                nll = nll.mean()
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers_weights[0][l_idx:r_idx]
                    bias_i = self.out_layers_biases[0][l_idx:r_idx]
                else:
                    weight_i = self.out_layers_weights[i]
                    bias_i = self.out_layers_biases[i]

                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], self.get_out_proj(0)

            logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            if target is not None:
                shift_logits = logit[..., :-1, :].contiguous()
                head_logprob = nn.functional.log_softmax(shift_logits, dim=1)
                target = target[..., 1:].contiguous()
                nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)

                offset = 0
                cutoff_values = [0] + self.cutoffs
                for i in range(len(cutoff_values) - 1):
                    l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                    mask_i = (target >= l_idx) & (target < r_idx)
                    indices_i = mask_i.nonzero(as_tuple=False).squeeze()

                    if indices_i.numel() == 0:
                        continue

                    target_i = target.index_select(0, indices_i) - l_idx
                    head_logprob_i = head_logprob.index_select(0, indices_i)

                    if i == 0:
                        logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                    else:
                        weight_i, bias_i, proj_i = weights[i], biases[i], self.get_out_proj(i)

                        hidden_i = hidden.index_select(0, indices_i)

                        tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                        tail_logprob_i = nn.functional.log_softmax(tail_logit_i, dim=1)

                        logprob_i = head_logprob_i[:, -i] + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1)

                    if self.keep_order or keep_order:
                        nll.index_copy_(0, indices_i, -logprob_i)
                    else:
                        nll[offset : offset + logprob_i.size(0)].copy_(-logprob_i)

                    offset += logprob_i.size(0)
                nll = nll.mean()

        return (logit, nll)  # TODO maybe cases for length or padding_mask


@dataclass
class S4ModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
    """

    last_hidden_state: torch.FloatTensor


@dataclass
class S4LMHeadModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        losses (:
            obj:`torch.FloatTensor` of shape `(batch_size, sequence_length-1)`, `optional`, returned when ``labels`` is
            provided) Language modeling losses (not reduced).
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token after SoftMax).
    """

    losses: Optional[torch.FloatTensor] = None
    prediction_scores: torch.FloatTensor = None

    @property
    def logits(self):
        return self.prediction_scores


class S4PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = S4Config
    base_model_prefix = "s4"

    def _init_weight_emb(self, weight, d: int, init_scale: Optional[float]):
        std = init_scale * (d ** -0.5)
        nn.init.normal_(weight, mean=0, std=std)

    def _init_bias(self, bias):
        nn.init.constant_(bias, 0.0)

    def _init_weights(self, module):
        """Initialize the weights"""
        classname = module.__class__.__name__
        if classname.find("Linear") != -1:
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        if classname.find("S4Embedding") != -1:
            if hasattr(module, "emb_projs"):
                for i in range(len(module.emb_projs)):
                    if module.emb_projs[i] is not None:
                        self._init_weight_emb(module.emb_projs[i], self.config.d_model, self.config.initializer_scale)
        elif classname.find("Embedding") != -1:
            if hasattr(module, "weight"):
                self._init_weight_emb(module.weight, module.embedding_dim, self.config.initializer_scale)
        elif classname.find("ProjectedAdaptiveLogSoftmax") != -1:
            if hasattr(module, "out_layers_biases"):
                bound = self.config.bias_scale * module.d_proj ** -0.5
                for i in range(len(module.out_layers_biases)):
                    nn.init.uniform_(module.out_layers_biases[i], -bound, bound)
        elif classname.find("LayerNorm") != -1:
            if hasattr(module, "weight"):
                nn.init.normal_(module.weight, 1.0, self.config.init_std)
            if hasattr(module, "bias") and module.bias is not None:
                self._init_bias(module.bias)

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, layer: Optional[int] = -1):
        """
        Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size. Take care of tying
        weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:

            new_num_tokens: (`optional`) int:
                New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at
                the end. Reducing the size will remove vectors from the end. If not provided or None: does nothing and
                just returns a pointer to the input tokens ``torch.nn.Embeddings`` Module of the model.
            layer: (`optional`) int:
                Layer of the `AdaptiveEmbedding` where the resizing should be done. Per default the last layer will be
                resized. Be aware that when resizing other than the last layer, you have to ensure that the new
                token(s) in the tokenizer are at the corresponding position.

        Return: ``torch.nn.Embeddings`` Pointer to the input tokens Embeddings Module of the model
        """
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed

        if new_num_tokens is None:
            return self.get_input_embeddings()

        new_num_tokens_layer, layer = self._get_new_num_tokens_layer(new_num_tokens, layer)
        assert new_num_tokens_layer > 0, "The size of the new embedding layer cannot be 0 or less"
        model_embeds = base_model._resize_token_embeddings(new_num_tokens_layer, layer)

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        base_model.vocab_size = new_num_tokens
        base_model.n_token = new_num_tokens

        new_embedding_shapes = self._get_embedding_shapes()
        self._resize_cutoffs(new_num_tokens, new_num_tokens_layer, new_embedding_shapes, layer)

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _get_new_num_tokens_layer(self, new_num_tokens, layer):
        embeddings = self.get_input_embeddings()
        if layer == -1:
            layer = len(embeddings.emb_layers) - 1
        assert 0 <= layer <= len(embeddings.emb_layers) - 1

        new_num_tokens_layer = (
            new_num_tokens
            - sum([emb.weight.shape[0] for emb in embeddings.emb_layers[:layer]])
            - sum([emb.weight.shape[0] for emb in embeddings.emb_layers[layer + 1 :]])
        )
        return new_num_tokens_layer, layer

    def _get_embedding_shapes(self):
        embeddings = self.get_input_embeddings()
        return [emb.weight.shape[0] for emb in embeddings.emb_layers]

    def _resize_token_embeddings(self, new_num_tokens, layer=-1):
        embeddings = self.get_input_embeddings()
        if new_num_tokens is None:
            return embeddings
        new_embeddings_layer = self._get_resized_embeddings(embeddings.emb_layers[layer], new_num_tokens)
        embeddings.emb_layers[layer] = new_embeddings_layer

        self.set_input_embeddings(embeddings)

        return self.get_input_embeddings()

    def _resize_cutoffs(self, new_num_tokens, new_emb_size, new_embedding_shapes, layer):
        embeddings = self.get_input_embeddings()

        for i in range(layer, len(embeddings.cutoffs)):
            embeddings.cutoffs[i] = sum(new_embedding_shapes[: i + 1])

        embeddings.cutoff_ends = [0] + embeddings.cutoffs
        embeddings.n_token = new_num_tokens

        self.config.cutoffs = embeddings.cutoffs[:-1]

        return embeddings.cutoffs


S4_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config (:class:`~transformers.S4Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

S4_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.S4Tokenizer`. See
            :func:`transformers.PreTrainedTokenizer.encode` and :func:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare S4 Model transformer outputting raw hidden-states without any specific head on top.",
    S4_START_DOCSTRING,
)
class S4Model(S4PreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config: S4Config):
        super().__init__(config)
        self.config = config

        self.embeddings = S4Embedding(config)

        self.d_model = config.d_model
        self.transposed = config.transposed

        if config.input_dropout_prob > 0.0:
            self.drop = (
                nn.Dropout2d(config.input_dropout_prob) if self.transposed else nn.Dropout(config.input_dropout_prob)
            )
        else:
            self.drop = None

        s4_config = {
            "d_state": config.d_state,
            "measure": config.measure,
            "rank": config.rank,
            "dt_min": config.dt_min,
            "dt_max": config.dt_max,
            "trainable": config.trainable_s4_params,
            "lr": config.learning_rate_s4_params,
            "cache": config.cache,
            "weight_decay": config.weight_decay,
            "l_max": config.l_max,
            "activation": config.activation_function,
            "postact": config.post_activation_function,
            "dropout": config.s4_dropout,
            "transposed": config.transposed,
        }
        if config.l_max is None:
            # correct length for S4
            s4_config["length_correction"] = True

        ff_config = {
            "expand": config.ff_expand,
            "activation": config.ff_activation_function,
            "transposed": config.transposed,
            "dropout": config.ff_dropout,
        }

        # Instantiate layers
        _layers = []
        d = config.d_model
        for i in range(1, (config.num_hidden_layers * 3) + 1):
            if i % 3 == 1:
                layer = S4(d, **s4_config)
            elif i % 3 == 2:
                layer = S4(d, **s4_config)
            else:
                layer = FF(d, **ff_config)

            block = SequenceResidualBlock(
                d,
                i_layer=i,
                prenorm=config.pre_norm,
                dropout=config.hidden_dropout_prob,
                layer=layer,
                residual=config.residual_connections,
                norm=config.normalize_type,
                pool_size=config.pool_size,
                pool_expand=config.pool_expand,
            )
            _layers.append(block)
            d = block.d_output

        self.d_output = d
        self.layers = nn.ModuleList(_layers)

        if config.pre_norm:
            if config.normalize_type is None:
                self.norm = None
            elif isinstance(config.normalize_type, str):
                self.norm = Normalization(self.d_output, transposed=config.transposed, _name_=config.normalize_type)
            else:
                self.norm = Normalization(self.d_output, transposed=config.transposed, **config.normalize_type)
        else:
            self.norm = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    @add_start_docstrings_to_model_forward(S4_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=S4ModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        state=None,
        return_dict=True,
    ):

        hidden_states = self.embeddings(input_ids)

        if self.transposed:
            hidden_states = einops.rearrange(hidden_states, "b l d -> b d l")
        if self.drop is not None:
            hidden_states = self.drop(hidden_states)

        prev_states = [None] * len(self.layers) if state is None else state
        next_states = []
        for i, (layer, prev_state) in enumerate(zip(self.layers, prev_states), start=1):
            hidden_states, state = layer(hidden_states, state=prev_state)  # TODO handle state
            next_states.append(state)
        hidden_states = self.norm(hidden_states)

        if self.transposed:
            hidden_states = einops.rearrange(hidden_states, "b d l -> b l d")

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                ]
                if v is not None
            )
        return S4ModelOutput(
            last_hidden_state=hidden_states,
        )


@add_start_docstrings("""S4 Model with a `language modeling` head on top for CLM fine-tuning.""", S4_START_DOCSTRING)
class S4LMHeadModel(S4PreTrainedModel):
    def __init__(self, config: S4Config):
        super().__init__(config)

        self.s4 = S4Model(config)
        if config.tie_weights:
            emb_layers = [i.weight for i in self.s4.embeddings.emb_layers]
        else:
            emb_layers = None

        emb_projs = self.s4.embeddings.emb_projs

        self.cls = ProjectedAdaptiveLogSoftmax(
            config.vocab_size,
            config.d_embed,
            config.d_model,
            config.cutoffs,
            div_val=config.div_val,
            tie_projs=config.tie_projs,
            out_layers_weights=emb_layers,
            out_projs=emb_projs,
            dropout=config.softmax_dropout_prob,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def tie_weights(self):
        """
        Run this to be sure output and input (adaptive) softmax weights are tied
        """
        if self.config.tie_weights:
            for i in range(len(self.cls.out_layers_weights)):
                self._tie_or_clone_weights(
                    self.cls.out_layers_weights[i],
                    self.s4.embeddings.emb_layers[i],
                )
        if self.config.tie_projs:
            for i in range(len(self.cls.shared_out_projs)):
                self.cls.shared_out_projs[i] = self.s4.embeddings.emb_projs[i]

    @add_start_docstrings_to_model_forward(S4_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=S4LMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        labels=None,
        return_dict=None,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`): Labels for
            computing the left-to-right language modeling loss (next word prediction). Indices should be in ``[-100, 0,
            ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.s4(
            input_ids,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        l_output = sequence_output.shape[1]
        sequence_output = sequence_output[..., -l_output:, :]
        prediction_scores, lm_loss = self.cls(sequence_output, labels)
        prediction_scores = prediction_scores.reshape(-1, l_output, prediction_scores.shape[-1])

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return S4LMHeadModelOutput(
            losses=lm_loss,
            prediction_scores=prediction_scores,
        )
