# TD [2023-01-05]: Copied from https://github.com/HazyResearch/state-spaces/blob/06dbbdfd0876501a7f12bf3262121badbc7658af/src/models/functional/vandermonde.py
# We add the interface to the log vandermonde CUDA code

"""pykeops implementations of the Vandermonde matrix multiplication kernel used in the S4D kernel."""

import torch

from einops import rearrange
from opt_einsum import contract
from pykeops.torch import Genred, LazyTensor


try:
    from cauchy_mult import vand_log_mult_sym_bwd, vand_log_mult_sym_fwd
except Exception:
    vand_log_mult_sym_fwd, vand_log_mult_sym_bwd = None, None


def _conj(x):
    return torch.cat([x, x.conj()], dim=-1)


def _broadcast_dims(*tensors):
    max_dim = max([len(tensor.shape) for tensor in tensors])
    tensors = [tensor.view((1,) * (max_dim - len(tensor.shape)) + tensor.shape) for tensor in tensors]
    return tensors


def _c2r(x):
    return torch.view_as_real(x)


def _r2c(x):
    return torch.view_as_complex(x)


def vandermonde_naive(v, x, L, conj=True):
    """
    v: (..., N) x: (..., N) returns: (..., L) \sum v x^l
    """
    if conj:
        x = _conj(x)
        v = _conj(v)
    vandermonde_matrix = x.unsqueeze(-1) ** torch.arange(L).to(x)  # (... N L)
    vandermonde_prod = torch.sum(v.unsqueeze(-1) * vandermonde_matrix, dim=-2)  # (... L)
    return vandermonde_prod


def log_vandermonde_naive(v, x, L, conj=True):
    """
    v: (..., N) x: (..., N) returns: (..., L) \sum v x^l
    """
    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x))  # (... N L)
    vandermonde_prod = contract("... n, ... n l -> ... l", v, vandermonde_matrix)  # (... L)
    if conj:
        return 2 * vandermonde_prod.real
    else:
        return vandermonde_prod


def log_vandermonde_lazy(v, x, L, conj=True):
    if conj:
        v = _conj(v)
        x = _conj(x)
    l = torch.arange(L).to(x)
    v, x, l = _broadcast_dims(v, x, l)
    v_l = LazyTensor(rearrange(v, "... N -> ... N 1 1"))
    x_l = LazyTensor(rearrange(x, "... N -> ... N 1 1"))
    l_l = LazyTensor(rearrange(l, "... L -> ... 1 L 1"))
    # exp
    vand = (x_l * l_l).exp()
    s = (v_l * vand).sum(dim=len(v_l.shape) - 2)
    return s.squeeze(-1)


def log_vandermonde(v, x, L, conj=True):
    expr = "ComplexMult(v, ComplexExp(ComplexMult(x, l)))"
    vandermonde_mult = Genred(
        expr,
        [
            "v = Vj(2)",
            "x = Vj(2)",
            "l = Vi(2)",
        ],
        reduction_op="Sum",
        axis=1,
    )

    l = torch.arange(L).to(x)
    v, x, l = _broadcast_dims(v, x, l)
    v = _c2r(v)
    x = _c2r(x)
    l = _c2r(l)

    r = vandermonde_mult(v, x, l, backend="GPU")
    if conj:
        return 2 * _r2c(r).real
    else:
        return _r2c(r)


def log_vandermonde_transpose_naive(u, v, x, L):
    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x))  # (... N L)
    vandermonde_prod = contract("... l, ... n, ... n l -> ... n", u.to(x), v.to(x), vandermonde_matrix)  # (... L)
    return vandermonde_prod


def log_vandermonde_transpose(u, v, x, L):
    """
    u: ... H L v: ... H N x: ... H N Returns: ... H N

    V = Vandermonde(a, L) : (H N L) contract_L(V * u * v)
    """
    expr = "ComplexMult(ComplexMult(v, u), ComplexExp(ComplexMult(x, l)))"
    vandermonde_mult = Genred(
        expr,
        [
            "u = Vj(2)",
            "v = Vi(2)",
            "x = Vi(2)",
            "l = Vj(2)",
        ],
        reduction_op="Sum",
        axis=1,
    )

    l = torch.arange(L).to(x)
    u, v, x, l = _broadcast_dims(u, v, x, l)
    u = _c2r(u)
    v = _c2r(v)
    x = _c2r(x)
    l = _c2r(l)

    r = vandermonde_mult(u, v, x, l, backend="GPU")
    return _r2c(r)


def _log_vandermonde_matmul(x, L):
    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x))  # (... N L)
    return vandermonde_matrix


def log_vandermonde_matmul(v, K):
    prod = contract("...n, ...nl -> ...l", v, K)
    return 2 * prod.real


class LogVandMultiplySymmetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, x, L):
        batch, N = v.shape
        supported_N_values = [1 << log_n for log_n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        if N not in supported_N_values:
            raise NotImplementedError(f"Only support N values in {supported_N_values}")
        max_L_value = 32 * 1024 * 64 * 1024
        if L > max_L_value:
            raise NotImplementedError(f"Only support L values <= {max_L_value}")
        if not v.is_cuda and x.is_cuda:
            raise NotImplementedError("Only support CUDA tensors")
        ctx.save_for_backward(v, x)
        return vand_log_mult_sym_fwd(v, x, L)

    @staticmethod
    def backward(ctx, dout):
        v, x = ctx.saved_tensors
        dv, dx = vand_log_mult_sym_bwd(v, x, dout)
        return dv, dx, None


if vand_log_mult_sym_fwd and vand_log_mult_sym_bwd is not None:
    log_vandermonde_fast = LogVandMultiplySymmetric.apply
else:
    log_vandermonde_fast = None
