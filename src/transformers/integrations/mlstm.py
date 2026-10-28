# Copyright 2025 NXAI GmbH. All rights reserved.
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
"""Native PyTorch mLSTM kernels, shared by xLSTM and Bolmo."""

from collections.abc import Callable

import torch
import torch.nn.functional as F


def soft_cap(values: torch.Tensor, cap_value: float | torch.Tensor | None = None) -> torch.Tensor:
    """
    Soft caps a tensor to a value.

    Performs a tanh operation on the logits and scales the result to the cap value. Common technique in attention
    and output language heads to prevent large logits from dominating the softmax. See for example Gemma2:
    https://huggingface.co/papers/2408.00118

    Args:
        values: The tensor to cap.
        cap_value: The value to cap the values to. If None, no cap is applied.

    Returns:
        The capped values.
    """
    if cap_value is None:
        return values
    return cap_value * torch.tanh(values / cap_value)


def mlstm_chunkwise_recurrent_fw_C(
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecB: torch.Tensor,
    vecI: torch.Tensor,
    matC_states: torch.Tensor | None = None,
    vecN_states: torch.Tensor | None = None,
    scaMinter_states: torch.Tensor | None = None,
    matC_initial: torch.Tensor | None = None,
    vecN_initial: torch.Tensor | None = None,
    scaMinter_initial: torch.Tensor | None = None,
    qk_scale: float | None = None,
    chunk_size: int = 64,
    num_chunks: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, nh, _, dhqk, dhhv = *matK.shape, matV.shape[-1]
    nc = num_chunks
    _dtype, _device = matK.dtype, matK.device

    if qk_scale is None:
        qk_scale = dhqk**-0.5

    # initialize the states tensors
    if matC_states is None:
        matC_states = torch.zeros((batch_size, nh, (nc + 1) * dhqk, dhhv), dtype=_dtype, device=_device)
    if vecN_states is None:
        vecN_states = torch.zeros((batch_size, nh, (nc + 1) * dhqk), dtype=_dtype, device=_device)
    if scaMinter_states is None:
        scaMinter_states = torch.zeros((batch_size, nh, (nc + 1)), dtype=_dtype, device=_device)

    # assign the initial states to the running states
    matC_k = (
        torch.zeros((batch_size, nh, dhqk, dhhv), dtype=_dtype, device=_device)
        if matC_initial is None
        else matC_initial
    )
    vecN_k = (
        torch.zeros((batch_size, nh, dhqk), dtype=_dtype, device=_device) if vecN_initial is None else vecN_initial
    )
    scaM_inter_k = (
        torch.zeros((batch_size, nh, 1), dtype=_dtype, device=_device)
        if scaMinter_initial is None
        else scaMinter_initial
    )
    vecA = vecB[..., -1, None] - vecB + vecI
    scaG = vecB[..., -1]
    scaA_max = vecA.max(-1).values

    scaM_inter_k = scaM_inter_k.squeeze(-1)

    for key in range(0, num_chunks):
        # store the states from the previous iteration before updating them
        # in the first iteration, these are the initial states
        matC_states[:, :, key * dhqk : (key + 1) * dhqk, :] = matC_k
        vecN_states[:, :, key * dhqk : (key + 1) * dhqk] = vecN_k
        scaMinter_states[:, :, key] = scaM_inter_k

        # m_k update
        scaA_max_k = scaA_max[:, :, key]
        scaG_k = scaG[:, :, key]
        scaM_inter_k_next = torch.max(scaG_k + scaM_inter_k, scaA_max_k)
        # C_k update
        matK_chunk = matK[:, :, key * chunk_size : (key + 1) * chunk_size, :]  # * qk_scale
        matV_chunk = matV[:, :, key * chunk_size : (key + 1) * chunk_size, :]
        vecA_k = vecA[:, :, key, :]

        vecAbar_k = torch.exp(vecA_k - scaM_inter_k_next[..., None])[:, :, :, None]

        matK_chunk_gated = matK_chunk * vecAbar_k

        scaGbar_k = torch.exp(scaG_k + scaM_inter_k - scaM_inter_k_next)[:, :, None]

        # NOTE: no update in-place (i.e. +=) as this gives error for autograd backward
        matC_k_next = scaGbar_k[..., None] * matC_k + matK_chunk_gated.transpose(-2, -1) @ (matV_chunk)

        # n_k update
        vecN_k_next = scaGbar_k * vecN_k + matK_chunk_gated.transpose(-2, -1).sum(-1)

        # move to the next iteration
        scaM_inter_k = scaM_inter_k_next
        matC_k = matC_k_next
        vecN_k = vecN_k_next

    # store the states from the last iteration
    matC_states[:, :, -dhqk:, :] = matC_k
    vecN_states[:, :, -dhqk:] = vecN_k
    scaMinter_states[:, :, -1] = scaM_inter_k

    return matC_states, vecN_states, scaMinter_states


def mlstm_chunkwise_parallel_fw_H(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    # these states must be all states up to the last chunk, i.e. :-1
    matC_states: torch.Tensor,
    vecN_states: torch.Tensor,
    scaMinter_states: torch.Tensor,
    vecI: torch.Tensor,
    vecB: torch.Tensor,
    qk_scale: float,
    chunk_size: int = 64,
    num_chunks: int = 1,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _device = matQ.device
    nc = num_chunks
    batch_size, nh, dqk, dhv = matC_states.shape
    dhqk = dqk // nc
    matC_k_states = matC_states.view(batch_size, nh, nc, dhqk, dhv)
    vecN_k_states = vecN_states.view(batch_size, nh, nc, dhqk)
    scaMinter_k_states = scaMinter_states

    matQ = matQ.view(batch_size, nh, nc, chunk_size, dhqk)
    matK = matK.view(batch_size, nh, nc, chunk_size, dhqk)
    matV = matV.view(batch_size, nh, nc, chunk_size, dhv)

    ltr = torch.tril(
        torch.ones(
            (chunk_size, chunk_size),
            dtype=torch.bool,
            device=_device,
        )
    )

    # Compute intra chunk contribution: H_intra
    matF_logsig_chunk = vecB[:, :, :, :, None] - vecB[:, :, :, None, :]

    matF_logsig_mask_chunk = torch.where(ltr, matF_logsig_chunk, -float("inf"))

    matLogD_chunk = matF_logsig_mask_chunk + vecI[:, :, :, None, :]

    # max_state intra
    vecMintra_k = torch.max(matLogD_chunk, dim=-1, keepdim=False).values

    # max_state combined
    vecM_b_inter = vecB + scaMinter_k_states[:, :, :, None]
    vecM_k_combine = torch.maximum(vecM_b_inter, vecMintra_k)

    vecM_k_combine = vecM_k_combine[:, :, :, :, None]
    vecM_b_inter = vecM_b_inter[:, :, :, :, None]

    matLogD_stabilized_chunk = matLogD_chunk - vecM_k_combine
    matD_chunk = torch.exp(matLogD_stabilized_chunk)

    matS_chunk = (matQ @ matK.transpose(-2, -1)) * qk_scale

    matM_chunk = matS_chunk * matD_chunk

    # ? Combine H_intra with H_inter
    vecBbar = torch.exp(vecM_b_inter - vecM_k_combine)
    matQ_chunk_gated = matQ * vecBbar * qk_scale

    matNumerator_common = matQ_chunk_gated @ matC_k_states + matM_chunk @ matV

    vecDenom_l_common = matQ_chunk_gated @ vecN_k_states.unsqueeze(-1) + matM_chunk.sum(dim=-1, keepdim=True)

    vecDenom_max_common = torch.maximum(torch.abs(vecDenom_l_common), torch.exp(-vecM_k_combine))

    matH_k_chunk = matNumerator_common / (vecDenom_max_common + eps)

    matH_out = matH_k_chunk.view(batch_size, nh, nc * chunk_size, dhv)

    # we need the denominator and the overall max state for the backward pass
    vecN_out = vecDenom_max_common.reshape(batch_size, nh, nc * chunk_size)
    vecM_out = vecM_k_combine.reshape(batch_size, nh, nc * chunk_size)
    return matH_out, vecN_out, vecM_out


def mlstm_chunkwise_fw(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    igate: torch.Tensor,
    fgate: torch.Tensor,
    cstate: torch.Tensor | None = None,
    nstate: torch.Tensor | None = None,
    mstate: torch.Tensor | None = None,
    qk_scale: float | None = None,
    return_last_states: bool = False,
    return_all_states: bool = False,
    chunk_size: int = 64,
    eps: float = 1e-6,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
]:
    batch_size, nh, sequence_length, dhqk = query.shape
    if sequence_length % chunk_size != 0:
        raise ValueError(f"Sequence length {sequence_length} is not divisible by chunk size {chunk_size}.")
    nc = sequence_length // chunk_size

    vecI = igate.view(batch_size, nh, nc, chunk_size)
    vecF = fgate.view(batch_size, nh, nc, chunk_size)

    # compute the gates, the g and the a and b vectors
    vecF_logsig = fgate.logsigmoid(vecF)
    vecB = vecF_logsig.cumsum(-1)

    if qk_scale is None:
        qk_scale = dhqk**-0.5

    #! materialize the  C_k, n_k, m_k states for each chunk
    matC_k_states, vecN_k_states, scaMinter_k_states = mlstm_chunkwise_recurrent_fw_C(
        matK=key,
        matV=value,
        vecB=vecB,
        vecI=vecI,
        matC_initial=cstate,
        vecN_initial=nstate,
        scaMinter_initial=mstate,
        qk_scale=qk_scale,
        chunk_size=chunk_size,
        num_chunks=nc,
    )

    #! compute the outputs within each chunk
    matH_out, vecN_out, vecM_out = mlstm_chunkwise_parallel_fw_H(
        matQ=query,
        matK=key,
        matV=value,
        matC_states=matC_k_states[:, :, :-dhqk, :],
        vecN_states=vecN_k_states[:, :, :-dhqk],
        scaMinter_states=scaMinter_k_states[:, :, :-1],
        vecI=vecI,
        vecB=vecB,
        qk_scale=qk_scale,
        chunk_size=chunk_size,
        num_chunks=nc,
        eps=eps,
    )

    ret_tuple = (matH_out, vecN_out, vecM_out)
    if return_last_states:
        ret_tuple += ((matC_k_states[:, :, -dhqk:, :], vecN_k_states[:, :, -dhqk:], scaMinter_k_states[:, :, -1:]),)
    else:
        ret_tuple += (None,)

    if return_all_states:
        ret_tuple += ((matC_k_states, vecN_k_states, scaMinter_k_states),)
    else:
        ret_tuple += (None,)

    return ret_tuple


def mlstm_chunkwise_native_autograd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    igate: torch.Tensor,
    fgate: torch.Tensor,
    c_initial: torch.Tensor | None = None,
    n_initial: torch.Tensor | None = None,
    m_initial: torch.Tensor | None = None,
    return_last_states: bool = False,
    eps: float = 1e-6,
    chunk_size: int = 64,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    batch_size, nh, sequence_length, dhqk = query.shape
    if sequence_length % chunk_size != 0:
        raise ValueError(f"Sequence length {sequence_length} is not divisible by chunk size {chunk_size}.")
    nc = sequence_length // chunk_size

    vecI = igate.view(batch_size, nh, nc, chunk_size)
    vecF = fgate.view(batch_size, nh, nc, chunk_size)

    # compute the gates, the g and the a and b vectors
    vecF_logsig = F.logsigmoid(vecF)
    vecB = vecF_logsig.cumsum(-1)

    qk_scale = dhqk**-0.5

    #! materialize the  C_k, n_k, m_k states for each chunk
    matC_k_states, vecN_k_states, scaMinter_k_states = mlstm_chunkwise_recurrent_fw_C(
        matK=key,
        matV=value,
        vecB=vecB,
        vecI=vecI,
        matC_initial=c_initial,
        vecN_initial=n_initial,
        scaMinter_initial=m_initial,
        qk_scale=qk_scale,
        chunk_size=chunk_size,
        num_chunks=nc,
    )

    #! compute the outputs within each chunk
    matH_out, vecN_out, vecM_out = mlstm_chunkwise_parallel_fw_H(
        matQ=query,
        matK=key,
        matV=value,
        matC_states=matC_k_states[:, :, :-dhqk, :],
        vecN_states=vecN_k_states[:, :, :-dhqk],
        scaMinter_states=scaMinter_k_states[:, :, :-1],
        vecI=vecI,
        vecB=vecB,
        qk_scale=qk_scale,
        chunk_size=chunk_size,
        num_chunks=nc,
        eps=eps,
    )

    last_states = (matC_k_states[:, :, -dhqk:, :], vecN_k_states[:, :, -dhqk:], scaMinter_k_states[:, :, -1:])

    if return_last_states:
        return matH_out, last_states
    else:
        return matH_out


def mlstm_recurrent_step_native(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    igate: torch.Tensor,
    fgate: torch.Tensor,
    cstate: torch.Tensor,
    nstate: torch.Tensor,
    mstate: torch.Tensor,
    eps: float = 1e-6,
    dtype_state: torch.dtype = torch.float32,
    **kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """This is a single step of the mLSTM operation in recurrent form."""
    dtype_qkv = query.dtype
    matC_old = cstate.to(dtype=dtype_state)
    vecN_old = nstate.to(dtype=dtype_state)
    scaM_old = mstate.to(dtype=dtype_state)

    batch_size, nh, dhqk = query.shape
    _, _, dhhv = value.shape
    if query.shape != key.shape:
        raise ValueError("query and key must have the same shape")
    if matC_old.shape != (batch_size, nh, dhqk, dhhv):
        raise ValueError(f"matC_old has wrong shape, got {matC_old.shape}")
    if vecN_old.shape != (batch_size, nh, dhqk):
        raise ValueError(f"vecN_old has wrong shape, got {vecN_old.shape}")
    if scaM_old.shape != (batch_size, nh, 1):
        raise ValueError(f"scaM_old has wrong shape, got {scaM_old.shape}")
    if igate.shape != (batch_size, nh, 1):
        raise ValueError(f"scaI has wrong shape, got {igate.shape}")
    if fgate.shape != (batch_size, nh, 1):
        raise ValueError(f"scaF has wrong shape, got {fgate.shape}")

    # gates
    scaF_log = torch.nn.functional.logsigmoid(fgate)

    # update rule
    scaM_state_new = torch.max(scaF_log + scaM_old, igate)

    scaF_act = torch.exp(scaF_log + scaM_old - scaM_state_new)
    scaI_act = torch.exp(igate - scaM_state_new)

    vecQ_scaled = query * (dhqk ** (-0.5))
    matC_state_new = scaF_act[:, :, :, None] * matC_old.clone() + scaI_act[:, :, :, None] * (
        key[:, :, :, None] @ value[:, :, None, :]
    )
    vecN_state_new = scaF_act * vecN_old.clone() + scaI_act * key
    h_num = vecQ_scaled[:, :, None, :] @ matC_state_new.to(dtype=dtype_qkv)
    h_num = h_num.squeeze(2).to(dtype=dtype_state)

    qn_dotproduct = vecQ_scaled[:, :, None, :] @ vecN_state_new[:, :, :, None].to(dtype=dtype_qkv)
    qn_dotproduct = qn_dotproduct.squeeze(2)
    max_val = torch.exp(-scaM_state_new)
    h_denom = (torch.maximum(qn_dotproduct.abs(), max_val) + eps).to(dtype=dtype_state)
    h = h_num / h_denom

    h = h.to(dtype=dtype_qkv)
    matC_state_new = matC_state_new.to(dtype=dtype_state)
    vecN_state_new = vecN_state_new.to(dtype=dtype_state)
    scaM_state_new = scaM_state_new.to(dtype=dtype_state)
    return h, (matC_state_new, vecN_state_new, scaM_state_new)


def mlstm_recurrent_sequence_native(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    igate: torch.Tensor,
    fgate: torch.Tensor,
    c_initial: torch.Tensor | None = None,
    n_initial: torch.Tensor | None = None,
    m_initial: torch.Tensor | None = None,
    return_last_states: bool = False,
    eps: float = 1e-6,
    dtype_state: torch.dtype = torch.float32,
    **kwargs,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
]:
    batch_size, nh, sequence_length, dhqk = query.shape
    dhv = value.shape[-1]
    device = query.device

    if c_initial is not None:
        if n_initial is None or m_initial is None:
            raise ValueError("Initial states must be provided together.")
        if n_initial is None or m_initial is None:
            raise ValueError("Initial states must be provided together.")
        matC_state, vecN_state, vecM_state = (
            c_initial.to(dtype=dtype_state),
            n_initial.to(dtype=dtype_state),
            m_initial.to(dtype=dtype_state),
        )
    else:
        # memory state
        matC_state = torch.zeros((batch_size, nh, dhqk, dhv), dtype=dtype_state, device=device)
        # normalizer state
        vecN_state = torch.zeros((batch_size, nh, dhqk), dtype=dtype_state, device=device)
        # max state
        vecM_state = torch.zeros((batch_size, nh, 1), dtype=dtype_state, device=device)

    vecH_list = []
    for t in range(sequence_length):
        # gates
        vecF_t, vecI_t = fgate[:, :, t, None], igate[:, :, t, None]

        # projections
        vecQ_t, vecK_t, vecV_t = query[:, :, t, :], key[:, :, t, :], value[:, :, t, :]

        # step
        vecH, (matC_state, vecN_state, vecM_state) = mlstm_recurrent_step_native(
            cstate=matC_state,
            nstate=vecN_state,
            mstate=vecM_state,
            query=vecQ_t,
            key=vecK_t,
            value=vecV_t,
            igate=vecI_t,
            fgate=vecF_t,
            eps=eps,
            dtype_state=dtype_state,
            **kwargs,
        )
        vecH_list.append(vecH)

    matH = torch.stack(vecH_list, dim=-2)

    if return_last_states:
        return matH, (matC_state, vecN_state, vecM_state)
    else:
        return matH


def wrap_chunkwise_pad_zeros(
    mlstm_chunkwise_kernel: Callable,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    fgate: torch.Tensor,
    igate: torch.Tensor,
    c_initial: torch.Tensor | None = None,
    n_initial: torch.Tensor | None = None,
    m_initial: torch.Tensor | None = None,
    return_last_states: bool = False,
    eps: float = 1e-6,
    autocast_kernel_dtype: torch.dtype = torch.bfloat16,
    chunk_size: int = 64,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if return_last_states:
        raise ValueError(
            "We are padding zeros, so we cannot return last states,",
            "as they would be not the true last states.",
        )

    batch_size, nh, sequence_length, dhqk = query.shape
    S_unpadded = sequence_length
    # padding to chunk size for kernels
    if sequence_length % chunk_size != 0:
        S_padded = ((sequence_length + chunk_size - 1) // chunk_size) * chunk_size
        q_pad = query.new_zeros(batch_size, nh, S_padded, query.shape[3])
        k_pad = key.new_zeros(batch_size, nh, S_padded, key.shape[3])
        v_pad = value.new_zeros(batch_size, nh, S_padded, value.shape[3])
        i_pad = igate.new_zeros(batch_size, nh, S_padded)
        f_pad = fgate.new_zeros(batch_size, nh, S_padded)
        q_pad[:, :, :S_unpadded, :] = query
        k_pad[:, :, :S_unpadded, :] = key
        v_pad[:, :, :S_unpadded, :] = value
        i_pad[:, :, :S_unpadded] = igate
        f_pad[:, :, :S_unpadded] = fgate
    else:
        q_pad = query
        k_pad = key
        v_pad = value
        i_pad = igate
        f_pad = fgate

    matH = mlstm_chunkwise_kernel(
        query=q_pad,
        key=k_pad,
        value=v_pad,
        igate=i_pad,
        fgate=f_pad,
        c_initial=c_initial,
        n_initial=n_initial,
        m_initial=m_initial,
        return_last_states=return_last_states,
        eps=eps,
        autocast_kernel_dtype=autocast_kernel_dtype,
        chunk_size=chunk_size,
        **kwargs,
    )
    matH = matH[:, :, :S_unpadded, :]
    return matH


def wrap_chunkwise_arbitrary_sequence_length(
    mlstm_chunkwise_kernel: Callable,
    mlstm_sequence_kernel: Callable,
    mlstm_step_kernel: Callable,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    fgate: torch.Tensor,
    igate: torch.Tensor,
    c_initial: torch.Tensor | None = None,
    n_initial: torch.Tensor | None = None,
    m_initial: torch.Tensor | None = None,
    return_last_states: bool = True,
    eps: float = 1e-6,
    autocast_kernel_dtype: torch.dtype = torch.bfloat16,
    chunk_size: int = 64,
    enable_logging: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """This function computes the last hidden state and matH outputs of the mLSTM, independently of the sequence length.

    For this it uses three kernels:
    - mlstm_chunkwise_kernel: mlstm chunkwise kernels that processes chunks of a given chunk size in parallel.
    - mlstm_sequence_kernel: mlstm kernel that processes the remaining sequence length in a single step recurrence.
    - mlstm_step_kernel: mlstm kernel that processes a sequence length of 1 in a single step.

    It tries to maximize the chunksizes to improve performance.
    It will start with the given chunk size and then divides the chunksize by 2 until the chunk size is smaller than 16.
    At every chunksize it will process the maximal number of chunks that fit into the remaining sequence length.

    E.g. for chunk_size = 64, this function will try the chunksizes [64, 32, 16] if necessary.

    For the remaining sequence length, which is smaller than 16, we use a different kernel that computes the mLSTM
    in a single step and loop over this in pytorch.

    Args:
        mlstm_chunkwise_kernel: The mLSTM chunkwise kernel that processes chunks of a given chunk size in parallel
        mlstm_sequence_kernel: The mLSTM kernel that processes the remaining sequence length in a single step recurrence
        query: The query tensor (batch_size, nh, sequence_length, dhqk)
        key: The key tensor (batch_size, nh, sequence_length, dhqk)
        value: The value tensor (batch_size, nh, sequence_length, dhhv)
        fgate: The forget gate tensor (batch_size, nh, sequence_length)
        igate: The input gate tensor (batch_size, nh, sequence_length)
        c_initial: The initial cell state tensor (batch_size, nh, dhqk, dhhv)
        n_initial: The initial hidden state tensor (batch_size, nh, dhqk)
        m_initial: The initial memory state tensor (batch_size, nh, 1)
        return_last_states: If True, the function will return the last states of the mLSTM
        eps: The epsilon value used for numerical stability
        autocast_kernel_dtype: The dtype used for the kernel computation
        chunk_size: The chunk size used for the chunkwise kernel
        enable_logging: If True, the function will log debug information. Default is False.

    Returns:
        The last hidden state tensor (batch_size, nh, sequence_length, dhhv) or a tuple containing the last hidden state tensor and the last states of the mLSTM
        Last states are (cstate (batch_size, nh, dhqk, dhhv), nstate (batch_size, nh, dhqk), mstate (batch_size, nh, 1)).
    """

    batch_size, nh, sequence_length, dhqk = key.shape
    dhhv = value.shape[-1]

    c_state = (
        c_initial
        if c_initial is not None
        else torch.zeros(batch_size, nh, dhqk, dhhv, device=key.device, dtype=torch.float32)
    )
    n_state = (
        n_initial
        if n_initial is not None
        else torch.zeros(batch_size, nh, dhqk, device=key.device, dtype=torch.float32)
    )
    m_state = (
        m_initial if m_initial is not None else torch.zeros(batch_size, nh, 1, device=key.device, dtype=torch.float32)
    )

    if sequence_length > 1:
        # process the sequence length in chunks
        h_outs = []
        seq_len_start_idx = 0
        remaining_seq_len = sequence_length - seq_len_start_idx
        num_chunks = remaining_seq_len // chunk_size
        if num_chunks > 0:
            iter_seq_len = chunk_size * num_chunks
            seq_len_idx = seq_len_start_idx + iter_seq_len
            h_out, (c_state, n_state, m_state) = mlstm_chunkwise_kernel(
                query=query[..., seq_len_start_idx:seq_len_idx, :].contiguous(),
                key=key[..., seq_len_start_idx:seq_len_idx, :].contiguous(),
                value=value[..., seq_len_start_idx:seq_len_idx, :].contiguous(),
                fgate=fgate[..., seq_len_start_idx:seq_len_idx].contiguous(),
                igate=igate[..., seq_len_start_idx:seq_len_idx].contiguous(),
                c_initial=c_state,
                n_initial=n_state,
                m_initial=m_state,
                chunk_size=chunk_size,
                return_last_states=True,
                autocast_kernel_dtype=autocast_kernel_dtype,
                eps=eps,
            )
            seq_len_start_idx += iter_seq_len
            h_outs.append(h_out)

        remaining_seq_len = sequence_length - seq_len_start_idx

        if remaining_seq_len > 0:
            # we use here matK as query as this kernel does not need a query, since we do not care about the outputs only about the last state
            h_out, (c_state, n_state, m_state) = mlstm_sequence_kernel(
                query=query[..., seq_len_start_idx:sequence_length, :].contiguous(),
                key=key[..., seq_len_start_idx:sequence_length, :].contiguous(),
                value=value[..., seq_len_start_idx:sequence_length, :].contiguous(),
                igate=igate[..., seq_len_start_idx:sequence_length].contiguous(),
                fgate=fgate[..., seq_len_start_idx:sequence_length].contiguous(),
                c_initial=c_state,
                n_initial=n_state,
                m_initial=m_state,
                return_last_states=True,
                eps=eps,
            )
            h_outs.append(h_out)
        h_out = torch.concatenate(h_outs, dim=2)

    else:
        if sequence_length != 1:
            raise ValueError(
                f"Received empty sequence (sequence_length={sequence_length}), require at least single element in the sequence."
            )
        # process the sequence length in a single step
        # while this case is also captured by the regular mode above,
        # it avoids the overhead of the loop and calls the step kernel directly
        # The step function does not want a sequence dimension
        # qkv shape is (batch_size, nh, dhqk/dhv)
        # igate, fgate shape is (batch_size, nh, 1)
        h_out, (c_state, n_state, m_state) = mlstm_step_kernel(
            query=query.squeeze(2),
            key=key.squeeze(2),
            value=value.squeeze(2),
            igate=igate,
            fgate=fgate,
            cstate=c_state,
            nstate=n_state,
            mstate=m_state,
            eps=eps,
        )
        h_out = h_out[:, :, None, :]

    if return_last_states:
        return h_out, (c_state, n_state, m_state)
    else:
        return h_out
