# TD: [2023-01-05]: Extracted the SSKernel class from
# https://github.com/HazyResearch/state-spaces/blob/06dbbdfd0876501a7f12bf3262121badbc7658af/src/models/sequence/ss/kernel.py
# We add option to use the shift kernel, and remove the option of SSKernelNPLR

"""SSM convolution kernels. SSKernel wraps different kernels with common options and handles the initialization.
"""

import logging
import math

import torch
import torch.nn as nn

from transformers.models.h3.src.models import dplr
from transformers.models.h3.src.models.ss_kernel_diag import EMAKernel, SSKernelDiag
from transformers.models.h3.src.models.ss_kernel_shift import SSKernelShift
from transformers.models.h3.src.ops.krylov import power


# from src.utils.utils import get_logger
# log = get_logger(__name__)

log = logging.getLogger()


def _conj(x):
    return torch.cat([x, x.conj()], dim=-1)


class SSKernel(nn.Module):
    """Wrapper around SSKernel parameterizations.

    The SSKernel is expected to support the interface forward() default_state() _setup_step() step()
    """

    def __init__(
        self,
        H,
        N=64,
        L=None,
        measure="diag-lin",
        rank=1,
        channels=1,
        dt_min=0.001,
        dt_max=0.1,
        deterministic=False,
        lr=None,
        mode="diag",
        n_ssm=None,
        verbose=False,
        measure_args={},
        **kernel_args,
    ):
        """State Space Kernel which computes the convolution kernel $\\bar{K}$

        H: Number of independent SSM copies; controls the size of the model. Also called d_model in the config. N:
        State size (dimensionality of parameters A, B, C). Also called d_state in the config. Generally shouldn't need
        to be adjusted and doens't affect speed much. L: Maximum length of convolution kernel, if known. Should work in
        the majority of cases even if not known. measure: Options for initialization of (A, B). For NPLR mode,
        recommendations are "legs", "fout", "hippo" (combination of both). For Diag mode, recommendations are
        "diag-inv", "diag-lin", "diag-legs", and "diag" (combination of diag-inv and diag-lin) rank: Rank of low-rank
        correction for NPLR mode. Needs to be increased for measure "legt" channels: C channels turns the SSM from a
        1-dim to C-dim map; can think of it having C separate "heads" per SSM. This was partly a feature to make it
        easier to implement bidirectionality; it is recommended to set channels=1 and adjust H to control parameters
        instead dt_min, dt_max: min and max values for the step size dt (\Delta) mode: Which kernel algorithm to use.
        'nplr' is the full S4 model; 'diag' is the simpler S4D; 'slow' is a dense version for testing n_ssm: Number of
        independent trainable (A, B) SSMs, e.g. n_ssm=1 means all A/B parameters are tied across the H different
        instantiations of C. n_ssm=None means all H SSMs are completely independent. Generally, changing this option
        can save parameters but doesn't affect performance or speed much. This parameter must divide H lr: Passing in a
        number (e.g. 0.001) sets attributes of SSM parameers (A, B, dt). A custom optimizer hook is needed to configure
        the optimizer to set the learning rates appropriately for these parameters.
        """
        super().__init__()
        self.N = N
        self.H = H
        dtype, cdtype = torch.float, torch.cfloat
        self.channels = channels
        self.n_ssm = n_ssm if n_ssm is not None else H
        self.mode = mode
        self.verbose = verbose
        self.kernel_args = kernel_args

        # Generate dt
        if deterministic:
            log_dt = torch.exp(torch.linspace(math.log(dt_min), math.log(dt_max), H))
        else:
            log_dt = torch.rand(self.H, dtype=dtype) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)

        # Compute the preprocessed representation
        if mode == "ema":
            self.kernel = EMAKernel(H, N=N, channels=channels, **kernel_args)
        else:
            w, P, B, V = dplr.combination(measure, self.N, rank, self.n_ssm, **measure_args)

            # Broadcast C to have H channels
            if deterministic:
                C = torch.zeros(channels, self.n_ssm, self.N, dtype=cdtype)
                C[:, :, :1] = 1.0
                C = torch.einsum("hmn, chn -> chm", V.conj().transpose(-1, -2), C)  # V^* C
                # C = repeat(C, "c t n -> c (v t) n", v=self.n_ssm // C.size(-2)).clone().contiguous()
                C = torch.flatten(C.unsqueeze(1).repeat(1, self.n_ssm // C.size(-2), 1, 1), 1, 2).clone().contiguous()
            else:
                C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype)

            # Broadcast other parameters to have n_ssm copies
            assert self.n_ssm % B.size(-2) == 0 and self.n_ssm % P.size(-2) == 0 and self.n_ssm % w.size(-2) == 0
            # Broadcast tensors to n_ssm copies
            # These will be the parameters, so make sure tensors are materialized and contiguous
            # B = repeat(B, "t n -> (v t) n", v=self.n_ssm // B.size(-2)).clone().contiguous()
            # P = repeat(P, "r t n -> r (v t) n", v=self.n_ssm // P.size(-2)).clone().contiguous()
            # w = repeat(w, "t n -> (v t) n", v=self.n_ssm // w.size(-2)).clone().contiguous()
            B = torch.flatten(B.unsqueeze(0).repeat(self.n_ssm // B.size(-2), 1, 1), 0, 1).clone().contiguous()
            P = torch.flatten(P.unsqueeze(1).repeat(1, self.n_ssm // P.size(-2), 1, 1), 1, 2).clone().contiguous()
            w = torch.flatten(w.unsqueeze(0).repeat(self.n_ssm // w.size(-2), 1, 1), 0, 1).clone().contiguous()

            if mode == "diag":
                if not measure.startswith("diag"):
                    log.warning(
                        "Diagonal kernel (S4D) activated but initialization is not intended for S4D. Set `measure` to"
                        " 'diag-lin', 'diag-inv', or 'diag-legs' for the main variants, or 'diag' for a combination of"
                        " S4D-Lin and S4D-Inv."
                    )
                # C = C * repeat(B, "t n -> (v t) n", v=H // self.n_ssm)
                C = C * torch.flatten(B.unsqueeze(0).repeat(H // self.n_ssm, 1, 1), 0, 1)
                self.kernel = SSKernelDiag(
                    w,
                    B,
                    C,
                    log_dt,
                    L=L,
                    lr=lr,
                    **kernel_args,
                )
            elif mode == "shift":
                # Initializing B to be e_1
                B = torch.zeros(self.H, self.N)
                B[..., 0] = 1.0
                # Match torch.Conv1d init
                C = torch.randn(self.H, self.channels, self.N)
                nn.init.kaiming_uniform_(C, a=math.sqrt(5))
                # C = rearrange(C, "h c n -> c h n")
                C = torch.permute(C, (1, 0, 2))
                self.kernel = SSKernelShift(B, C, L=L, lr=lr, **kernel_args)
            else:
                raise NotImplementedError(f"{mode=} is not valid")

    def forward(self, state=None, L=None, rate=None):
        return self.kernel(state=state, L=L, rate=rate)

    @torch.no_grad()
    def forward_state(self, u, state):
        """Forward the state through a sequence, i.e. computes the state after passing chunk through SSM

        state: (B, H, N) u: (B, H, L)

        Returns: (B, H, N)
        """

        if hasattr(self.kernel, "forward_state"):
            return self.kernel.forward_state(u, state)

        dA, dB = self.kernel._setup_state()  # Construct dA, dB matrices
        # dA, dB = self.kernel.dA, self.kernel.dB # (H N N) (H N)

        conj = state.size(-1) != dA.size(-1)
        if conj:
            state = _conj(state)

        v = torch.einsum("h n, b h l -> b h n l", dB, u.flip(-1))  # dB.unsqueeze(-1) * u.flip(-1).unsqueeze(-2)
        AL, v = power(u.size(-1), dA, v)
        next_state = torch.einsum("h m n, b h n -> b h m", AL, state)
        next_state = next_state + v

        if conj:
            next_state = next_state[..., : next_state.size(-1) // 2]
        return next_state

    def _setup_step(self, **kwargs):
        # This method is intended to be private so that setting up an S4 module with
        # ```
        # if hasattr(module, 'setup_step'): module.setup_step()
        # ```
        # will not trigger this method multiple times
        self.kernel._setup_step(**kwargs)

    def step(self, u, state, **kwargs):
        y, state = self.kernel.step(u, state, **kwargs)
        return y, state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)
