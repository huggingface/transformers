import torch
import torch.autograd
import torch.utils.cpp_extension
import os
from dataclasses import dataclass
from typing import Union
cvmm_module = None


def load_cvmm():
    global cvmm_module
    if cvmm_module is None:
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'cvmm.cu')

        old_flags = torch.utils.cpp_extension.COMMON_NVCC_FLAGS
        # This hack is needed because certain versions nvcc fails to compile if the flag is present
        new_flags = [f for f in old_flags if f != '--expt-relaxed-constexpr']
        print(f"Hacking nvcc flags from {old_flags} to {new_flags}")
        torch.utils.cpp_extension.COMMON_NVCC_FLAGS = new_flags

        cvmm_module = torch.utils.cpp_extension.load(
            'cvmm', [filename], verbose=True)

        torch.utils.cpp_extension.COMMON_NVCC_FLAGS = old_flags


def counts_and_offsets(sel, n_experts):
    cnts = torch.zeros(n_experts, dtype=torch.int32, device=sel.device)
    cnts.index_add_(0, sel, torch.ones_like(sel))
    offsets = cnts.cumsum(0, dtype=sel.dtype)
    return cnts, offsets


@dataclass
class CVMMSel:
    raw_sel: torch.Tensor
    sel: torch.Tensor
    sel_index: torch.Tensor
    cnts: torch.Tensor
    offsets: torch.Tensor


def cvmm_prepare_sel(sel: torch.Tensor, n_experts: int) -> CVMMSel:
    load_cvmm()

    rsel = sel
    sel = sel.flatten()

    assert sel.is_contiguous()
    ssel, sindex = cvmm_module.sort_indices(sel, n_experts)
    sel_cnts, sel_offsets = counts_and_offsets(sel, n_experts)

    return CVMMSel(rsel, ssel, sindex, sel_cnts, sel_offsets)


class CVMM(torch.autograd.Function):
    warned = False

    @staticmethod
    def forward(ctx, x: torch.Tensor, sel: Union[torch.Tensor, CVMMSel], keys: torch.Tensor):
        if not isinstance(sel, CVMMSel):
            sel = cvmm_prepare_sel(sel, keys.shape[0])

        xorig = x
        x = x.flatten(end_dim=-2)

        assert x.is_contiguous()
        assert sel.sel.is_contiguous()
        assert keys.is_contiguous() or (keys.stride(1) == 1 and keys.stride(2) == keys.shape[1])
        if not (x.shape[-1] % 4 == 0 and keys.shape[-1] % 4 == 0):
            print(f"Both x and keys must be divisible by 4. Shapes: x: {x.shape}, keys: {keys.shape}")
            assert False

        if (not CVMM.warned) and not (x.shape[-1] % 64 == 0 and keys.shape[-1] % 64 == 0):
            CVMM.warned = True
            print(f"CVMM is the fastest if both x and keys must be divisible by 64. Shapes: x: {x.shape}, keys: {keys.shape}")

        res = cvmm_module.cvmm_sorted_blocktile_co_raw(x, sel.sel, keys, sel.sel_index, sel.cnts, sel.offsets)

        ctx.save_for_backward(xorig, keys, sel.sel, sel.sel_index, sel.cnts, sel.offsets)
        return res.view(*xorig.shape[:-1], keys.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_contiguous()

        xorig, keys, ssel, sindex, sel_cnts, sel_offsets = ctx.saved_tensors

        x = xorig.flatten(end_dim=-2)
        grad_output = grad_output.flatten(end_dim=-2)

        grad_x = cvmm_module.cvmm_sorted_blocktile_co_raw(grad_output, ssel, keys.transpose(1, 2), sindex, sel_cnts, sel_offsets)
        grad_keys = cvmm_module.cvmm_sorted_blocktile_co_raw_project_grads(keys.shape[0], x, ssel, grad_output, sindex, sel_cnts, sel_offsets)

        return grad_x.view_as(xorig), None, grad_keys


cvmm = CVMM.apply

if __name__ == "__main__":
    from torch.autograd import gradcheck

    n_experts = 4
    n_channels = 64
    expert_size = 64
    bs = 32

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32
    atol_tresh = 1e-2

    keys = torch.nn.Parameter(torch.randn(n_experts, n_channels, expert_size, dtype=dtype, device=device))
    testvec = torch.randn(bs, n_channels, dtype=dtype, device=device)
    sel = torch.randint(0, n_experts, (bs,), dtype=torch.int32, device=device)
    test_grad = torch.randn(bs, expert_size, dtype=dtype, device=device)

    olist = []
    for b in range(bs):
        olist.append(testvec[b:b+1] @ keys[sel[b]])
    ref = torch.cat(olist, dim=0)

    out = cvmm(testvec, sel, keys)
    assert torch.allclose(ref, out, atol=atol_tresh, rtol=0)

    print("Forward ok.")

    keys = keys.requires_grad_(True)
    testvec = testvec.requires_grad_(True)

    assert gradcheck(cvmm, (testvec, sel, keys), eps=1e-2, atol=1e-4)
    print("Backward ok.")
