"""Ring attention implementation using flash attention adapted from https://github.com/zhuzilin/ring-flash-attention/"""
import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward,
    _flash_attn_varlen_forward,
)
from torch.distributed.device_mesh import init_device_mesh
import os
from typing import Optional, Tuple



def ring_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # This is before the transpose
    seq_len = query.shape[2]

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2).squeeze(0)
    key = key.transpose(1, 2).squeeze(0)
    value = value.transpose(1, 2).squeeze(0)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)

    from torch.distributed.device_mesh import init_device_mesh
    import os
    tp_size = int(os.getenv("TP_SIZE", 1))
    cp_size = int(os.getenv("CP_SIZE", 1))
    device_mesh = init_device_mesh("cuda", (tp_size, cp_size), mesh_dim_names=("tp","cp")) #TODO: this is just a placeholder, we need to get the actual device_mesh
    process_group = device_mesh.get_group("cp")

    cu_seqlens = torch.arange(0, seq_len + 1, seq_len // tp_size, dtype=torch.int32, device=query.device) #TODO: this is just a placeholder, we need to get the actual cu_seqlens
    max_seqlen = seq_len // cp_size #TODO: this is just a placeholder, we need to get the actual max_seqlen

    attn_output = ring_flash_attn_varlen_func(
        query.to(torch.bfloat16).squeeze(0), # TODO: not sure why query is casted to float32
        key.to(torch.bfloat16).squeeze(0),
        value.to(torch.bfloat16).squeeze(0),
        cu_seqlens,
        max_seqlen,
        scaling=scaling,
        dropout=dropout,
        causal=module.is_causal if hasattr(module, "is_causal") else True,
        window_size=sliding_window if sliding_window is not None else (-1, -1),
        alibi_slopes=None,
        ring_pg=process_group,
    )[0]
    return attn_output, None

def ring_flash_attn_varlen_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens,
    max_seqlen,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    comm = RingComm(process_group)

    out = None
    lse = None
    next_k, next_v = None, None

    old_lse = False
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)
        if not causal or step <= comm.rank:
            params = get_default_args(_flash_attn_varlen_forward).copy()
            params.update(
                {
                    "q": q,
                    "k": k,
                    "v": v,
                    "cu_seqlens_q": cu_seqlens,
                    "cu_seqlens_k": cu_seqlens,
                    "max_seqlen_q": max_seqlen,
                    "max_seqlen_k": max_seqlen,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": causal and step == 0,
                    "alibi_slopes": alibi_slopes,
                    "return_softmax": True and dropout_p > 0,
                }
            )
            if "window_size" in params:
                params.update({"window_size": window_size})
            else:
                params.update(
                    {
                        "window_size_left": window_size[0],
                        "window_size_right": window_size[1],
                    }
                )

            outputs = _flash_attn_varlen_forward(**params)
            if len(outputs) == 8:
                block_out, _, _, _, _, block_lse, _, _ = outputs
            else:
                assert len(outputs) == 4
                block_out, block_lse, _, _ = outputs
            if block_lse.dim() == 3:
                old_lse = True
                block_lse = flatten_varlen_lse(
                    block_lse,
                    cu_seqlens=cu_seqlens,
                )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    if old_lse:
        lse = unflatten_varlen_lse(lse, cu_seqlens, max_seqlen)
    else:
        lse = lse.squeeze(dim=-1).transpose(0, 1)
    return out, lse


def ring_flash_attn_varlen_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    cu_seqlens,
    max_seqlen,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None

    block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    next_dk, next_dv = None, None
    next_k, next_v = None, None
    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        if step <= kv_comm.rank or not causal:
            bwd_causal = causal and step == 0
            params = get_default_args(_flash_attn_varlen_backward).copy()
            params.update(
                {
                    "dout": dout,
                    "q": q,
                    "k": k,
                    "v": v,
                    "out": out,
                    "softmax_lse": softmax_lse,
                    "dq": block_dq_buffer,
                    "dk": block_dk_buffer,
                    "dv": block_dv_buffer,
                    "cu_seqlens_q": cu_seqlens,
                    "cu_seqlens_k": cu_seqlens,
                    "max_seqlen_q": max_seqlen,
                    "max_seqlen_k": max_seqlen,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": bwd_causal,
                    "alibi_slopes": alibi_slopes,
                    "deterministic": deterministic,
                }
            )
            if "window_size" in params:
                params.update({"window_size": window_size})
            else:
                params.update(
                    {
                        "window_size_left": window_size[0],
                        "window_size_right": window_size[1],
                    }
                )
            _flash_attn_varlen_backward(**params)

            if dq is None:
                dq = block_dq_buffer.to(torch.float32)
                dk = block_dk_buffer.to(torch.float32)
                dv = block_dv_buffer.to(torch.float32)
            else:
                dq += block_dq_buffer
                d_kv_comm.wait()
                dk = block_dk_buffer + next_dk
                dv = block_dv_buffer + next_dv
        elif step != 0:
            d_kv_comm.wait()
            dk, dv = next_dk, next_dv

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv)

    d_kv_comm.wait()

    return dq.to(torch.bfloat16), next_dk.to(q.dtype), next_dv.to(q.dtype)


class RingFlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = ring_flash_attn_varlen_forward(
            group,
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens = ctx.saved_tensors
        dq, dk, dv = ring_flash_attn_varlen_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens,
            ctx.max_seqlen,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def ring_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens=None,
    max_seqlen=None,
    dropout=0.0,
    scaling=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    ring_pg=None,
    **kwargs,
):
    assert cu_seqlens is not None
    assert max_seqlen is not None
    return (
        RingFlashAttnVarlenFunc.apply(
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            dropout,
            scaling,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
            ring_pg,
        ),
    )


## triton_utils.py
import torch
import triton
import triton.language as tl


@triton.jit
def flatten_kernel(
    # pointers to matrices
    OUT,
    LSE,
    CU_SEQLENS,
    # strides
    stride_out_nheads,
    stride_out_seqlen,
    stride_lse_batch,
    stride_lse_nheads,
    stride_lse_seqlen,
    # meta-parameters
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    start_idx = tl.load(CU_SEQLENS + pid_batch)
    seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
    LSE = LSE + pid_batch * stride_lse_batch + pid_head * stride_lse_nheads
    OUT = OUT + pid_head * stride_out_nheads + start_idx * stride_out_seqlen

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    LSE = LSE + rm[:, None] * stride_lse_seqlen
    x = tl.load(LSE, mask=rm[:, None] < seqlen, other=0.0)

    OUT = OUT + rm[:, None] * stride_out_seqlen
    tl.store(OUT, x, mask=rm[:, None] < seqlen)


def flatten_varlen_lse(lse, cu_seqlens):
    """
    Arguments:
        lse: (batch_size, nheads, max_seqlen)
        cu_seqlens: (batch_size + 1,)
    Return:
        flatten_lse: (nheads, total_seqlen)
    """
    total_seqlen = cu_seqlens[-1]
    batch_size, nheads, max_seqlen = lse.shape
    output = torch.empty((nheads, total_seqlen), dtype=lse.dtype, device=lse.device)

    def grid(META):
        return triton.cdiv(max_seqlen, META["BLOCK_M"]), batch_size, nheads

    BLOCK_M = 4

    with torch.cuda.device(lse.device.index):
        flatten_kernel[grid](
            output,
            lse,
            cu_seqlens,
            # strides
            output.stride(0),
            output.stride(1),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            BLOCK_M,
        )
    return output


@triton.jit
def unflatten_kernel(
    # pointers to matrices
    OUT,
    LSE,
    CU_SEQLENS,
    # strides
    stride_out_batch,
    stride_out_nheads,
    stride_out_seqlen,
    stride_lse_seqlen,
    stride_lse_nheads,
    # meta-parameters
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    start_idx = tl.load(CU_SEQLENS + pid_batch)
    seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
    LSE = LSE + pid_head * stride_lse_nheads + start_idx * stride_lse_seqlen
    OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    LSE = LSE + rm[:, None] * stride_lse_seqlen
    x = tl.load(LSE, mask=rm[:, None] < seqlen, other=0.0)

    OUT = OUT + rm[:, None] * stride_out_seqlen
    tl.store(OUT, x, mask=rm[:, None] < seqlen)


def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    """
    Arguments:
        lse: (total_seqlen, nheads, 1)
        cu_seqlens: (batch_size + 1,)
        max_seqlen: int
    Return:
        unflatten_lse: (batch_size, nheads, max_seqlen)
    """
    lse = lse.unsqueeze(dim=-1)
    batch_size = len(cu_seqlens) - 1
    nheads = lse.shape[1]
    output = torch.empty(
        (batch_size, nheads, max_seqlen),
        dtype=lse.dtype,
        device=lse.device,
    )

    def grid(META):
        return triton.cdiv(max_seqlen, META["BLOCK_M"]), batch_size, nheads

    BLOCK_M = 4

    with torch.cuda.device(lse.device.index):
        unflatten_kernel[grid](
            output,
            lse,
            cu_seqlens,
            # strides
            output.stride(0),
            output.stride(1),
            output.stride(2),
            lse.stride(0),
            lse.stride(1),
            BLOCK_M,
        )
    return output


import inspect
from functools import cache
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

__all__ = ["update_out_and_lse", "RingComm", "get_default_args"]


@cache
def _get_default_args(func):
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None,) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if "softcap" in args:
        args["softcap"] = 0.0
    return args


def get_default_args(func):
    if inspect.isfunction(func):
        return _get_default_args(func)
    else:
        # Use the origin _init_fn in CustomOpDef
        return _get_default_args(func._init_fn)


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


@torch.jit.script
def flatten_varlen_lse(lse, cu_seqlens):
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)


@torch.jit.script
def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    num_seq = len(cu_seqlens) - 1
    num_head = lse.shape[-2]
    new_lse = torch.empty((num_seq, max_seqlen, num_head, 1), dtype=torch.float32, device=lse.device)
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse[i, : end - start] = lse[start:end]
    return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous()


## utils.py
class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_k, next_v = self.send_recv(k, k_buffer), self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v


class AllGatherComm:
    def __init__(self, group=None) -> None:
        self.group = group
        self.handles = []

    def all_gather(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor):
        handle = dist.all_gather_into_tensor(output_tensor, input_tensor, group=self.group, async_op=True)
        self.handles.append(handle)

    def wait(self):
        for handle in self.handles:
            handle.wait()
        self.handles = []
