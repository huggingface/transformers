# TD: [2023-01-05]: Extracted the SSKernelDiag class from
# https://github.com/HazyResearch/state-spaces/blob/06dbbdfd0876501a7f12bf3262121badbc7658af/src/models/sequence/ss/kernel.py
# We make a small change to use the log_vandermonde CUDA code.

"""SSKernelDiag is the S4D kernel, a simpler algorithm for computing the kernel for the case of diagonal state matrices A.
"""
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from opt_einsum import contract
from transformers.models.h3.src.models.ssm_utils import OptimModule

# This could be None if the CUDA import fails
from transformers.models.h3.src.ops.vandermonde import log_vandermonde_fast


# from src.utils import get_logger
# logger = get_logger(__name__)

logger = logging.getLogger()


try:
    from transformers.models.h3.src.ops.vandermonde import log_vandermonde, log_vandermonde_transpose

    has_pykeops = True
    logger.info("Pykeops installation found.")
except ImportError:
    has_pykeops = False
    from transformers.models.h3.src.ops.vandermonde import log_vandermonde_naive as log_vandermonde
    from transformers.models.h3.src.ops.vandermonde import log_vandermonde_transpose_naive as log_vandermonde_transpose

    logger.warning("Falling back on slow Vandermonde kernel. Install pykeops for improved memory efficiency.")


_c2r = torch.view_as_real
_r2c = torch.view_as_complex

if tuple(map(int, torch.__version__.split(".")[:2])) >= (1, 10):

    def _resolve_conj(x):
        return x.conj().resolve_conj()

else:

    def _resolve_conj(x):
        return x.conj()


class SSKernelDiag(OptimModule):
    """Version using (complex) diagonal state matrix (S4D)"""

    def __init__(
        self,
        A,
        B,
        C,
        log_dt,
        L=None,
        disc="bilinear",
        real_type="exp",
        lr=None,
        bandlimit=None,
        force_real=False,
    ):

        super().__init__()
        self.L = L
        self.disc = disc
        self.bandlimit = bandlimit
        self.real_type = real_type
        self.force_real = force_real

        # Rank of low-rank correction
        assert A.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = A.size(-1)
        assert A.size(-2) == B.size(-2)  # Number of independent SSMs trained
        assert self.H % A.size(-2) == 0
        self.n_ssm = A.size(-2)
        self.repeat = self.H // A.size(0)

        self.channels = C.shape[0]
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))

        # Register parameters
        if lr is None or isinstance(lr, float):
            lr_dict = {}
        else:
            lr_dict, lr = lr, None

        self.register("log_dt", log_dt, lr_dict.get("dt", lr))
        self.register("B", _c2r(B), lr_dict.get("B", lr))
        self.register("inv_A_real", self._A_init(A.real), lr_dict.get("A", lr))
        self.register("A_imag", A.imag, lr_dict.get("A", lr))

    def _A_init(self, A_real):
        A_real = torch.clamp(A_real, max=-1e-4)
        if self.real_type == "none":
            return -A_real
        elif self.real_type == "exp":
            return torch.log(-A_real)  # Some of the HiPPO methods have real part 0
        elif self.real_type == "relu":
            return -A_real
        elif self.real_type == "sigmoid":
            return torch.logit(-A_real)
        elif self.real_type == "softplus":
            return torch.log(torch.exp(-A_real) - 1)
        else:
            raise NotImplementedError

    def _A(self):
        # Get the internal A (diagonal) parameter
        if self.real_type == "none":
            A_real = -self.inv_A_real
        elif self.real_type == "exp":
            A_real = -torch.exp(self.inv_A_real)
        elif self.real_type == "relu":
            # JAX version seems to NaN if you alloA 0's, although this code Aas fine Aithout it
            A_real = -F.relu(self.inv_A_real) - 1e-4
        elif self.real_type == "sigmoid":
            A_real = -F.sigmoid(self.inv_A_real)
        elif self.real_type == "softplus":
            A_real = -F.softplus(self.inv_A_real)
        else:
            raise NotImplementedError
        A = A_real + 1j * self.A_imag
        return A

    def forward(self, L, state=None, rate=1.0, u=None):
        """
        state: (B, H, N) initial state rate: sampling rate factor L: target length returns: (C, H, L) convolution
        kernel (generally C=1) (B, H, L) output from initial state
        """

        dt = torch.exp(self.log_dt) * rate  # (H)
        C = _r2c(self.C)  # (C H N)
        A = self._A()  # (H N)

        B = _r2c(self.B)
        B = repeat(B, "t n -> 1 (v t) n", v=self.repeat)

        # Force A to be real valued, so the whole kernel can be interpreted as a "multi-head EMA"
        if self.force_real:
            A = A.real + 0j

        if self.bandlimit is not None:
            freqs = dt[:, None] / rate * A.imag.abs() / (2 * math.pi)  # (H, N)
            mask = torch.where(freqs < self.bandlimit * 0.5, 1, 0)
            C = C * mask

        # Incorporate dt into A
        A = repeat(A, "t n -> (v t) n", v=self.repeat)
        dtA = A * dt.unsqueeze(-1)  # (H N)

        # Augment B with state
        if state is not None:
            s = state / dt.unsqueeze(-1)
            if self.disc == "bilinear":
                s = s * (1.0 + dtA / 2)
            elif self.disc == "zoh":
                s = s * dtA * dtA.exp() / (dtA.exp() - 1.0)
            B = torch.cat([s, B], dim=-3)  # (1+B H N)

        C = (B[:, None, :, :] * C).view(-1, self.H, self.N)
        if self.disc == "zoh":
            # Power up
            C = C * (torch.exp(dtA) - 1.0) / A
            # TODO (TD): make it work for C.shape[0] > 1
            if log_vandermonde_fast is not None and C.shape[0] == 1:
                K = log_vandermonde_fast(C.squeeze(0), dtA, L).unsqueeze(0)  # (H L)
            else:
                K = log_vandermonde(C, dtA, L)  # (H L)
        elif self.disc == "bilinear":
            C = C * (1.0 - dtA / 2).reciprocal() * dt.unsqueeze(-1)  # or * dtA / A
            dA = (1.0 + dtA / 2) / (1.0 - dtA / 2)
            if log_vandermonde_fast is not None:
                dA_log = repeat(dA.log(), "h d -> (c h) d", c=C.shape[0])
                K = rearrange(
                    log_vandermonde_fast(rearrange(C, "c h d -> (c h) d"), dA_log, L), "(c h) d -> c h d", c=C.shape[0]
                )
            else:
                K = log_vandermonde(C, dA.log(), L)
        elif self.disc == "dss":
            # Implementation from DSS meant for case when real eigenvalues can be positive
            P = dtA.unsqueeze(-1) * torch.arange(L, device=C.device)  # [H N L]
            A_gt_0 = A.real > 0  # [N]
            if A_gt_0.any():
                with torch.no_grad():
                    P_max = dtA * (A_gt_0 * (L - 1))  # [H N]
                P = P - P_max.unsqueeze(-1)  # [H N L]
            S = P.exp()  # [H N L]

            dtA_neg = dtA * (1 - 2 * A_gt_0)  # [H N]
            num = dtA_neg.exp() - 1  # [H N]
            den = (dtA_neg * L).exp() - 1  # [H N]

            # Inline reciprocal function for DSS logic
            x = den * A
            x_conj = _resolve_conj(x)
            r = x_conj / (x * x_conj + 1e-7)

            C = C * num * r  # [C H N]
            K = contract("chn,hnl->chl", C, S).float()
        else:
            assert False, f"{self.disc} not supported"

        K = K.view(-1, self.channels, self.H, L)  # (1+B C H L)
        if state is not None:
            K_state = K[:-1, :, :, :]  # (B C H L)
        else:
            K_state = None
        K = K[-1, :, :, :]  # (C H L)
        return K, K_state

    def _setup_step(self):
        # These methods are organized like this to be compatible with the NPLR kernel interface
        dt = torch.exp(self.log_dt)  # (H)
        B = _r2c(self.B)  # (H N)
        C = _r2c(self.C)  # (C H N)
        self.dC = C
        A = self._A()  # (H N)

        A = repeat(A, "t n -> (v t) n", v=self.repeat)
        B = repeat(B, "t n -> (v t) n", v=self.repeat)

        # Incorporate dt into A
        dtA = A * dt.unsqueeze(-1)  # (H N)
        if self.disc == "zoh":
            self.dA = torch.exp(dtA)  # (H N)
            self.dB = B * (torch.exp(dtA) - 1.0) / A  # (C H N)
        elif self.disc == "bilinear":
            self.dA = (1.0 + dtA / 2) / (1.0 - dtA / 2)
            self.dB = B * (1.0 - dtA / 2).reciprocal() * dt.unsqueeze(-1)  # or * dtA / A

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        next_state = contract("h n, b h n -> b h n", self.dA, state) + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        return 2 * y.real, next_state

    def forward_state(self, u, state):
        self._setup_step()
        AL = self.dA ** u.size(-1)
        u = u.flip(-1).to(self.dA).contiguous()  # (B H L)
        v = log_vandermonde_transpose(u, self.dB, self.dA.log(), u.size(-1))
        next_state = AL * state + v
        return next_state


class EMAKernel(OptimModule):
    """Translation of Mega's MultiHeadEMA.
    This is a minimal implementation of the convolution kernel part of the module. This module, together with the main
    S4 block in src.models.sequence.ss.s4 (which is really just a fft-conv wrapper around any convolution kernel, such
    as this one), should be exactly equivalent to using the original Mega EMA module in src.models.sequence.ss.ema. Two
    additional flags have been provided to resolve discrepencies in parameter count between S4(D) and EMA
    - `dt_tie` makes the shape of the step size \Delta (H, 1) instead of (H, N)
    - `efficient_bidirectional` ties the A/B/dt parameters for the conv kernels in both forwards and backwards
      directions. This should have exactly the same speed, slightly more parameter efficiency, and unchanged
      performance.
    """

    def __init__(
        self,
        H,
        N=2,
        channels=1,
        l_max=None,
        dt_tie=False,
        efficient_bidirectional=False,
    ):
        super().__init__()

        self.H = H
        self.N = N
        self.channels = channels
        self.l_max = l_max
        self.scale = math.sqrt(1.0 / self.N)

        # Exactly match the parameter count of S4(D) when bididirectional is on
        self.efficient_bidirectional = efficient_bidirectional
        if self.efficient_bidirectional:
            H_C = H * channels
        else:
            H *= channels
            H_C = H

        self.delta = nn.Parameter(torch.Tensor(H, 1 if dt_tie else N, 1))
        self.alpha = nn.Parameter(torch.Tensor(H, N, 1))
        self.beta = nn.Parameter(torch.Tensor(H, N, 1))
        self.gamma = nn.Parameter(torch.Tensor(H_C, N))
        # self.omega = nn.Parameter(torch.Tensor(H))  # D skip connection handled by outside class

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.normal_(self.delta, mean=0.0, std=0.2)
            nn.init.normal_(self.alpha, mean=0.0, std=0.2)
            # Mega comment: beta [1, -1, 1, -1, ...] seems more stable.
            val = torch.ones(self.N, 1)
            if self.N > 1:
                idx = torch.tensor(list(range(1, self.N, 2)))
                val.index_fill_(0, idx, -1.0)
            self.beta.normal_(mean=0.0, std=0.02).add_(val)
            nn.init.normal_(self.gamma, mean=0.0, std=1.0)
            # nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def coeffs(self):  # Same as discretize
        p = torch.sigmoid(self.delta)  # (H N 1)
        alpha = torch.sigmoid(self.alpha)
        q = 1.0 - p * alpha
        return p, q

    def forward(self, L=None, state=None, rate=1.0):
        L = L if self.l_max is None else min(self.l_max, L)
        p, q = self.coeffs()  # (H N 1)
        vander = torch.arange(L).to(p).view(1, 1, L) * torch.log(q)  # (H N L)
        kernel = (p * self.beta) * torch.exp(vander)
        if self.efficient_bidirectional:
            C = rearrange(self.gamma * self.scale, "(c h) n -> c h n", c=self.channels)
            kernel = torch.einsum("dnl,cdn->cdl", kernel, C)
            # kernel = rearrange(kernel, 'c d l -> (c d) l')
        else:
            kernel = torch.einsum("dnl,dn->dl", kernel, self.gamma * self.scale)
            kernel = rearrange(kernel, "(c h) l -> c h l", c=self.channels)

        kernel = kernel[..., :L]
        # kernel = rearrange(kernel, '(c h) l -> c h l', c=self.channels)
        return kernel, None  # k_state
