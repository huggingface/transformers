# TD: [2023-01-05]: Extracted the SSKernelDiag class from
# https://github.com/HazyResearch/state-spaces/blob/06dbbdfd0876501a7f12bf3262121badbc7658af/src/models/sequence/ss/kernel.py
# We make a small change to use the log_vandermonde CUDA code.

"""SSKernelDiag is the S4D kernel, a simpler algorithm for computing the kernel for the case of diagonal state matrices A.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.h3.src.models.ssm_utils import OptimModule


class SSKernelShift(OptimModule):
    def __init__(self, B, C, L=None, lr=None, **kwargs):
        """
        B: (H, d), real C: (channel, H, d), real
        """
        super().__init__()
        self.L = L
        self.N = B.size(-1)
        self.H = B.shape[0]

        # Register parameters
        if lr is None or isinstance(lr, float):
            lr_dict = {}
        else:
            lr_dict, lr = lr, None
        self.register("B", B, lr_dict.get("B", lr))
        self.C = nn.Parameter(C)

    def forward(self, state=None, rate=1.0, L=None):
        if L is None:
            L = self.L
        # This class doesn't support variable length functionalities, since it's a discrete SSM
        assert rate == 1.0 and L is not None

        # Augment B with state
        B = self.B
        if state is not None:
            B = torch.cat([B.unsqueeze(0), state], dim=-3).unsqueeze(1)  # (1 + B, 1, H, N)
        B_f = torch.fft.rfft(B, n=2 * self.N)
        C_f = torch.fft.rfft(self.C, n=2 * self.N)
        k = torch.fft.irfft(B_f.conj() * C_f, n=2 * self.N)[..., : min(self.N, L)]
        # If self.N < L, need to pad with zeros to reach length L
        if self.N < L:
            k = F.pad(k, (0, L - self.N))
        k = k.float()  # Otherwise it could be dtype half
        if state is not None:
            k, k_state = k[0], k[1:]
        else:
            k_state = None
        return k, k_state

    def _setup_step(self):
        # Just here to conform to the interface, eventually we should refactor out
        pass

    def default_state(self, *batch_shape):
        return torch.zeros(*batch_shape, self.H, self.N, dtype=self.C.dtype, device=self.C.device)

    def step(self, u, state):
        """u: (B, H), state: (B, H, N)"""
        next_state = F.pad(state, (1, -1)) + torch.einsum("h n, b h -> b h n", self.B, u)
        y = torch.einsum("c h n, b h n -> b c h", self.C, next_state)
        return y, next_state

    def forward_state(self, u, state):
        """u: (B, H, L), state: (B, H, N)"""
        L = u.shape[-1]
        B_f = torch.fft.rfft(self.B, n=2 * self.N)
        u_f = torch.fft.rfft(u[..., -self.N :].flip(-1).to(dtype=self.B.dtype), n=2 * self.N)
        v = torch.fft.irfft(B_f * u_f, n=2 * self.N)[..., : self.N]
        if L < self.N:
            next_state = F.pad(state, (L, -L)) + v
        else:
            next_state = v
        return next_state
