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
"""PyTorch xLSTM Model."""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...generation import GenerationMixin
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple, is_xlstm_available
from .configuration_xlstm import xLSTMConfig


if is_xlstm_available():
    from xlstm.xlstm_large.model import RMSNorm as xLSTMRMSNorm
    from xlstm.xlstm_large.model import mLSTMBlock as xLSTMBlock
    from xlstm.xlstm_large.model import mLSTMStateType, soft_cap

    external_xlstm = True
else:
    from functools import partial
    from typing import Callable, Literal

    from .configuration_xlstm import round_up_to_next_multiple_of

    mLSTMLayerStateType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    mLSTMStateType = dict[int, mLSTMLayerStateType]

    external_xlstm = False

    def soft_cap(values: torch.Tensor, cap_value: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
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
        matC_states: torch.Tensor = None,
        vecN_states: torch.Tensor = None,
        scaMinter_states: torch.Tensor = None,
        matC_initial: torch.Tensor = None,
        vecN_initial: torch.Tensor = None,
        scaMinter_initial: torch.Tensor = None,
        qk_scale: Optional[float] = None,
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
        nc, chunk_size = num_chunks, chunk_size
        batch_size, nh, dqk, dhv = matC_states.shape
        matC_k_states = matC_states.view(batch_size, nh, nc, dqk // nc, dhv)
        vecN_k_states = vecN_states.view(batch_size, nh, nc, dqk // nc)
        scaMinter_k_states = scaMinter_states

        matQ = matQ.view(batch_size, nh, nc, chunk_size, dqk)
        matK = matK.view(batch_size, nh, nc, chunk_size, dqk)
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
        vecM_out = vecM_k_combine(batch_size, nh, nc * chunk_size)
        return matH_out, vecN_out, vecM_out

    def mlstm_chunkwise_fw(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        igate: torch.Tensor,
        fgate: torch.Tensor,
        cstate: torch.Tensor = None,
        nstate: torch.Tensor = None,
        mstate: torch.Tensor = None,
        qk_scale: Optional[float] = None,
        return_last_states: bool = False,
        return_all_states: bool = False,
        chunk_size: int = 64,
        eps: float = 1e-6,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
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
            ret_tuple += (
                (matC_k_states[:, :, -dhqk:, :], vecN_k_states[:, :, -dhqk:], scaMinter_k_states[:, :, -1:]),
            )
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
        c_initial: torch.Tensor = None,
        n_initial: torch.Tensor = None,
        m_initial: torch.Tensor = None,
        return_last_states: bool = False,
        eps: float = 1e-6,
        chunk_size: int = 64,
        **kwargs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
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
        matC_state_new = scaF_act[:, :, :, None] * matC_old + scaI_act[:, :, :, None] * (
            key[:, :, :, None] @ value[:, :, None, :]
        )
        vecN_state_new = scaF_act * vecN_old + scaI_act * key
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
        c_initial: torch.Tensor = None,
        n_initial: torch.Tensor = None,
        m_initial: torch.Tensor = None,
        return_last_states: bool = False,
        eps: float = 1e-6,
        dtype_state: torch.dtype = torch.float32,
        **kwargs,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
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
        c_initial: torch.Tensor = None,
        n_initial: torch.Tensor = None,
        m_initial: torch.Tensor = None,
        return_last_states: bool = False,
        eps: float = 1e-6,
        autocast_kernel_dtype: torch.dtype = torch.bfloat16,
        chunk_size: int = 64,
        **kwargs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
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
        c_initial: torch.Tensor = None,
        n_initial: torch.Tensor = None,
        m_initial: torch.Tensor = None,
        return_last_states: bool = True,
        eps: float = 1e-6,
        autocast_kernel_dtype: torch.dtype = torch.bfloat16,
        chunk_size: int = 64,
        enable_logging: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
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
            m_initial
            if m_initial is not None
            else torch.zeros(batch_size, nh, 1, device=key.device, dtype=torch.float32)
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

    class xLSTMBackend(nn.Module):
        """xLSTM Backend Module for PyTorch.

        This module wraps the xLSTM kernels and provides a high-level interface for training and inference.
        """

        config_class = xLSTMConfig

        def __init__(self, config: xLSTMConfig):
            super().__init__()
            self.config = config
            self.chunkwise_kernel_fn = mlstm_chunkwise_native_autograd
            self.sequence_kernel_fn = mlstm_recurrent_sequence_native
            self.step_kernel_fn = mlstm_recurrent_step_native

            self._inference_fn = partial(
                wrap_chunkwise_arbitrary_sequence_length,
                mlstm_chunkwise_kernel=self.chunkwise_kernel_fn,
                mlstm_sequence_kernel=partial(
                    self.sequence_kernel_fn,
                    dtype_state=getattr(torch, config.inference_state_dtype),
                ),
                mlstm_step_kernel=partial(
                    self.step_kernel_fn,
                    dtype_state=getattr(torch, config.inference_state_dtype),
                ),
                chunk_size=config.chunk_size,
                eps=config.eps,
                autocast_kernel_dtype=getattr(torch, config.autocast_kernel_dtype),
                return_last_states=True,
            )

            train_kernel_fn = partial(
                self.chunkwise_kernel_fn,
                autocast_kernel_dtype=getattr(torch, config.autocast_kernel_dtype),
                eps=config.eps,
                chunk_size=config.chunk_size,
            )
            if "with_padding" in config.mode:
                train_kernel_fn = partial(wrap_chunkwise_pad_zeros, mlstm_chunkwise_kernel=train_kernel_fn)
            self._train_fn = train_kernel_fn

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            igate: torch.Tensor,
            fgate: torch.Tensor,
            c_initial: torch.Tensor = None,
            n_initial: torch.Tensor = None,
            m_initial: torch.Tensor = None,
            return_last_states: bool = False,
            mode: Optional[Literal["train", "inference"]] = None,
        ) -> Union[torch.Tensor, tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
            """Forward pass of the mLSTM backend.

            Depending on the configured mode, this method will call the appropriate kernel function.

            Args:
                query: The query tensor of shape (batch_size, nh, sequence_length, dhqk).
                key: The key tensor of shape (batch_size, nh, sequence_length, dhqk).
                value: The value tensor of shape (batch_size, nh, sequence_length, dhhv).
                igate: The input gate preactivation tensor of shape (batch_size, nh, sequence_length).
                fgate: The forget gate preactivation tensor of shape (batch_size, nh, sequence_length).
                c_initial: The initial cell state tensor of shape (batch_size, nh, dhqk, dhhv).
                                                    Defaults to None.
                n_initial: The initial hidden state tensor of shape (batch_size, nh, dhqk). Defaults to None.
                m_initial: The initial memory tensor of shape (batch_size, nh, 1). Defaults to None.
                return_last_states: Whether to return the last states of the sequence. Defaults to None.
                                                    If None, the value from the config is used.

            Returns:
                hidden states of shape (batch_size, nh, sequence_length, dhhv)
                hidden states and last states the last states are the cell state cstate (batch_size, nh, dhqk, dhhv),
                the normalizer state nstate (batch_size, nh, dhqk), and the max state mstate (batch_size, nh, 1)
            """
            if mode is None:
                mode = self.config.mode

            if "train" in mode:
                if return_last_states is None:
                    return_last_states = self.config.return_last_states

                if self.config.mode == "train_with_padding":
                    if return_last_states:
                        raise ValueError("return_last_states=True is not supported with train_with_padding mode.")

                return self._train_fn(
                    query=query,
                    key=key,
                    value=value,
                    igate=igate,
                    fgate=fgate,
                    c_initial=c_initial,
                    n_initial=n_initial,
                    m_initial=m_initial,
                    return_last_states=return_last_states,
                )

            elif "inference" in mode:
                # inference mode always returns the last states
                return self._inference_fn(
                    query=query,
                    key=key,
                    value=value,
                    igate=igate,
                    fgate=fgate,
                    c_initial=c_initial,
                    n_initial=n_initial,
                    m_initial=m_initial,
                )
            else:
                raise ValueError(f"Unknown mode: {self.config.mode}")

        def extra_repr(self) -> str:
            return f"{self.config}"

    class xLSTMRMSNorm(nn.Module):
        """Root mean square normalization layer implementation similar
        to https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html.

        It normalizes the input tensor by the root mean square of the last dimension.

        Args:
            num_features: The number of features in the input tensor.
            eps: A small value to avoid division by zero.
            use_weight: Whether to use a learnable weight.
            use_bias: Whether to use a learnable bias.
            force_float32_reductions: Whether to force float32 reductions.
        """

        def __init__(
            self,
            num_features: int,
            eps: float = 1e-6,
            use_weight: bool = True,
            use_bias: bool = False,
            force_float32_reductions: bool = True,
        ):
            super().__init__()
            self.num_features = num_features
            self.eps = eps
            self.force_float32_reductions = force_float32_reductions

            if use_weight:
                self.weight = nn.Parameter(torch.ones(num_features))
            else:
                self.weight = None

            if use_bias:
                self.bias = nn.Parameter(torch.zeros(num_features))
            else:
                self.bias = None

        def _apply_weight_bias(self, x: torch.Tensor) -> torch.Tensor:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
            return x

        def _rms_normalize(self, x: torch.Tensor) -> torch.Tensor:
            # apply rms norm over the last dimension, i.e. HD dimension
            in_dtype = x.dtype
            if self.force_float32_reductions:
                x = x.float()
            x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return x.to(in_dtype)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self._rms_normalize(x)
            x = self._apply_weight_bias(x)
            return x

    class xLSTMMultiHeadLayerNorm(nn.Module):
        """Multi-head version of the LayerNorm layer.

        It normalizes the last dimension of the input tensor.

        The input is assumed to have the shape (batch_size, sequence_length, nh, DH), where:
        batch_size: batch size
        sequence_length: sequence length
        nh: number of heads
        DH: head dimension

        The normalization is applied over the last dimension (DH) of the input tensor.

        Args:
            num_heads: The number of heads.
            head_dim: The head dimension.
            eps: A small value to avoid division by zero.
            use_weight: Whether to use a learnable weight.
            use_bias: Whether to use a learnable bias.
            force_float32_reductions: Whether to force float32 reductions

        Returns:
            The normalized tensor with the shape (batch_size, sequence_length, nh * DH).
        """

        def __init__(
            self,
            num_heads: int,
            head_dim: int,
            eps: float = 1e-6,
            use_weight: bool = True,
            use_bias: bool = False,
            force_float32_reductions: bool = True,
        ):
            super().__init__()
            self.num_features = num_heads * head_dim
            self.eps = eps
            self.force_float32_reductions = force_float32_reductions

            if use_weight:
                self.weight = nn.Parameter(torch.ones(self.num_features))
            else:
                self.weight = None

            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.num_features))
            else:
                self.bias = None
            self.num_heads = num_heads
            self.head_dim = head_dim

        def _apply_weight_bias(self, x: torch.Tensor) -> torch.Tensor:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
            return x

        def _layer_normalize(self, x: torch.Tensor) -> torch.Tensor:
            # apply layer norm over the last dimension, i.e. HD dimension
            in_dtype = x.dtype
            if self.force_float32_reductions:
                x = x.float()
            x_centered = x - x.mean(dim=-1, keepdim=True)
            y = x_centered * torch.rsqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
            return y.to(in_dtype)

        def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
            batch_size, sequence_length, nh, DH = x.shape
            if nh != self.num_heads:
                raise ValueError(f"Expected {self.num_heads} heads, got {nh}, input shape: {x.shape}")
            if DH != self.head_dim:
                raise ValueError(f"Expected {self.head_dim} head dimension, got {DH}, input shape: {x.shape}")

            x = self._layer_normalize(x)
            x = x.reshape(batch_size, sequence_length, -1)
            x = self._apply_weight_bias(x)
            return x

    class xLSTMFeedForward(nn.Module):
        def __init__(self, config: xLSTMConfig):
            super().__init__()
            self.config = config

            self.up_proj_dim = round_up_to_next_multiple_of(
                config.hidden_size * config.ffn_proj_factor,
                config.ffn_round_up_to_multiple_of,
            )

            if self.config.weight_mode == "single":
                self.proj_up_gate = nn.Linear(
                    in_features=config.hidden_size,
                    out_features=self.up_proj_dim,
                    bias=self.config.use_bias,
                )
                self.proj_up = nn.Linear(
                    in_features=config.hidden_size,
                    out_features=self.up_proj_dim,
                    bias=self.config.use_bias,
                )
            elif self.config.weight_mode == "fused":
                self.proj_up_gate_z = nn.Linear(
                    in_features=config.hidden_size,
                    out_features=2 * self.up_proj_dim,
                    bias=self.config.use_bias,
                )

            self.proj_down = nn.Linear(
                in_features=self.up_proj_dim,
                out_features=config.hidden_size,
                bias=self.config.use_bias,
            )

            self.act_fn = nn.SiLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.config.weight_mode == "single":
                x = self.act_fn(self.proj_up_gate(x)) * self.proj_up(x)
            elif self.config.weight_mode == "fused":
                x = self.proj_up_gate_z(x)
                gate, z = torch.tensor_split(x, (self.up_proj_dim,), dim=-1)
                x = self.act_fn(gate) * z

            y = self.proj_down(x)
            return y

    class xLSTMLayer(nn.Module):
        def __init__(self, config: xLSTMConfig):
            super().__init__()
            self.config = config

            self.v_dim = int(config.hidden_size * config.v_dim_factor)
            self.qk_dim = int(config.hidden_size * config.qk_dim_factor)

            if self.config.weight_mode == "single":
                self.q = nn.Linear(
                    in_features=self.config.hidden_size,
                    out_features=self.qk_dim,
                    bias=self.config.use_bias,
                )
                self.k = nn.Linear(
                    in_features=self.config.hidden_size,
                    out_features=self.qk_dim,
                    bias=self.config.use_bias,
                )
                self.v = nn.Linear(
                    in_features=self.config.hidden_size,
                    out_features=self.v_dim,
                    bias=self.config.use_bias,
                )

                self.ogate_preact = nn.Linear(
                    in_features=self.config.hidden_size,
                    out_features=self.v_dim,
                    bias=self.config.use_bias,
                )
                self.igate_preact = nn.Linear(
                    in_features=self.config.hidden_size,
                    out_features=self.config.num_heads,
                    bias=True,
                )
                self.fgate_preact = nn.Linear(
                    in_features=self.config.hidden_size,
                    out_features=self.config.num_heads,
                    bias=True,
                )
            elif self.config.weight_mode == "fused":
                self.qkv_opreact = nn.Linear(
                    in_features=self.config.hidden_size,
                    out_features=2 * self.qk_dim + 2 * self.v_dim,
                    bias=self.config.use_bias,
                )
                self.ifgate_preact = nn.Linear(
                    in_features=self.config.hidden_size,
                    out_features=2 * self.config.num_heads,
                    bias=True,
                )

            self.ogate_act_fn = nn.Sigmoid()
            self.mlstm_backend = xLSTMBackend(config=self.config)

            self.multihead_norm = xLSTMMultiHeadLayerNorm(
                num_heads=self.config.num_heads,
                head_dim=self.v_dim // self.config.num_heads,
                eps=self.config.norm_eps,
                use_weight=True,
                use_bias=self.config.use_bias,
                force_float32_reductions=self.config.norm_reduction_force_float32,
            )
            self.out_proj = nn.Linear(
                in_features=self.v_dim,
                out_features=self.config.hidden_size,
                bias=self.config.use_bias,
            )

        def forward(
            self, x: torch.Tensor, state: Optional[mLSTMLayerStateType] = None
        ) -> tuple[torch.Tensor, Optional[mLSTMLayerStateType]]:
            if x.ndim != 3:
                raise ValueError(f"Input must have shape [batch_size, sequence_length, HD], got {x.shape}")
            batch_size, sequence_length, _ = x.shape
            if self.config.weight_mode == "single":
                query = self.q(x)
                key = self.k(x)
                value = self.v(x)
                o_preact = self.ogate_preact(x)
                i_preact = soft_cap(self.igate_preact(x), cap_value=self.config.gate_soft_cap)
                f_preact = soft_cap(self.fgate_preact(x), cap_value=self.config.gate_soft_cap)

            elif self.config.weight_mode == "fused":
                qkv_opreact = self.qkv_opreact(x)
                query, key, value, o_preact = torch.tensor_split(
                    qkv_opreact,
                    (
                        self.qk_dim,
                        2 * self.qk_dim,
                        2 * self.qk_dim + self.v_dim,
                    ),
                    dim=-1,
                )

                if_preact = soft_cap(self.ifgate_preact(x), cap_value=self.config.gate_soft_cap)
                i_preact, f_preact = torch.tensor_split(if_preact, (self.config.num_heads,), dim=-1)

            query = query.reshape(batch_size, sequence_length, self.config.num_heads, -1).transpose(1, 2)
            key = key.reshape(batch_size, sequence_length, self.config.num_heads, -1).transpose(1, 2)
            value = value.reshape(batch_size, sequence_length, self.config.num_heads, -1).transpose(1, 2)
            i_preact = i_preact.transpose(1, 2)
            f_preact = f_preact.transpose(1, 2)
            if state is None:
                c_initial, n_initial, m_initial = None, None, None
            else:
                c_initial, n_initial, m_initial = state

            h, state = self.mlstm_backend(
                query=query,
                key=key,
                value=value,
                igate=i_preact,
                fgate=f_preact,
                c_initial=c_initial,
                n_initial=n_initial,
                m_initial=m_initial,
            )
            expected_h_shape = (
                batch_size,
                self.config.num_heads,
                sequence_length,
                self.v_dim // self.config.num_heads,
            )
            if h.shape != expected_h_shape:
                raise ValueError(f"Got {h.shape}, expected {expected_h_shape}")

            h = h.transpose(1, 2)
            h_norm = self.multihead_norm(h)
            h_norm = h_norm.reshape(batch_size, sequence_length, -1)

            h_out = self.ogate_act_fn(o_preact) * h_norm

            y = self.out_proj(h_out)
            return y, state

    class xLSTMBlock(nn.Module):
        def __init__(self, config: xLSTMConfig):
            super().__init__()
            self.config = config
            self.norm_mlstm = xLSTMRMSNorm(
                num_features=config.hidden_size,
                eps=config.norm_eps,
                use_weight=True,
                use_bias=config.use_bias,
                force_float32_reductions=config.norm_reduction_force_float32,
            )
            self.mlstm_layer = xLSTMLayer(config)
            self.norm_ffn = xLSTMRMSNorm(
                num_features=config.hidden_size,
                eps=config.norm_eps,
                use_weight=True,
                use_bias=config.use_bias,
                force_float32_reductions=config.norm_reduction_force_float32,
            )
            self.ffn = xLSTMFeedForward(config)

        def forward(
            self, x: torch.Tensor, state: Optional[mLSTMStateType] = None
        ) -> tuple[torch.Tensor, mLSTMStateType]:
            x_mlstm = self.norm_mlstm(x)
            x_mlstm, state = self.mlstm_layer(x_mlstm, state)
            x = x + x_mlstm

            x_ffn = self.norm_ffn(x)
            x_ffn = self.ffn(x_ffn)
            x = x + x_ffn

            return x, state


def small_init_method(dim):
    """
    Adapted from: https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py
    Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution."""
    std = (2 / (5 * dim)) ** (1 / 2)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def wang_init_method(n_layers, dim):
    """
    Adapted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py
    """
    std = 2 / n_layers / dim ** (1 / 2)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class xLSTMPreTrainedModel(PreTrainedModel):
    """
    An abstract class for an interface to loading a pre-trained xLSTM model.
    """

    config_class = xLSTMConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["xLSTMBlock"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _module_name_map(self, module):
        for name, mod in self.named_modules():
            if mod is module:
                return name
        return ""

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            small_init_method(self.config.hidden_size)(self.embeddings.weight)
        elif isinstance(module, nn.Linear):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if self.config.weight_mode == "single" and "gate" in self._module_name_map(module):
                torch.nn.init.zeros_(module.weight)
                with torch.no_grad():
                    if "igate" in self._module_name_map(module):
                        module.bias.copy_(-10.0 * torch.ones_like(module.bias))
                    elif "fgate" in self._module_name_map(module):
                        module.bias.copy_(
                            torch.linspace(
                                3.0,
                                6.0,
                                module.bias.shape[-1],
                            ).to(
                                device=module.bias.device,
                                dtype=module.bias.dtype,
                            )
                        )
            elif self.config.weight_mode == "fused" and "gate" in self._module_name_map(module):
                torch.nn.init.zeros_(module.weight)
                with torch.no_grad():
                    module.bias[: self.config.num_heads] += -module.bias[
                        : self.config.num_heads
                    ] - 10.0 * torch.ones_like(module.bias)
                    module.bias[: self.config.num_heads] += -module.bias[self.config.num_heads :] + torch.linspace(
                        3.0,
                        6.0,
                        module.bias.shape[-1],
                    ).to(
                        device=module.bias.device,
                        dtype=module.bias.dtype,
                    )
            elif "proj_down" in self._module_name_map(module):
                wang_init_method(dim=module.weight.shape[1], n_layers=self.config.num_hidden_layers)(module.weight)
            elif "out_proj" in self._module_name_map(module):
                wang_init_method(dim=self.config.hidden_size, n_layers=self.config.num_hidden_layers)(module.weight)
            elif module.weight is not None:
                small_init_method(self.config.hidden_size)(module.weight)
        elif isinstance(module, xLSTMRMSNorm) or hasattr(module, "_layer_normalize"):
            torch.nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)


class xLSTMCache:
    """
    Cache for xLSTM model which does not have attention mechanism and key value states.

    Arguments:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The batch size with which the model will be used.
        dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
            The default `dtype` to use when initializing the layer.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. Should be the same as the layer.

    Attributes:
        seqlen_offset: int
        dtype: torch.dtype

    Example:

        ```python
        >>> from transformers import AutoTokenizer, xLSTMForCausalLM, xLSTMCache

        >>> model = xLSTMForCausalLM.from_pretrained("NX-AI/xLSTM-7b")
        >>> tokenizer = xLSTMTokenizer.from_pretrained("NX-AI/xLSTM-7b")

        >>> inputs = tokenizer(text="I am an xLSTM", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> cache_params = xLSTMCache(config=model.config, max_batch_size=1, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, cache_params=cache_params, use_cache=True)
        >>> outputs.cache_params
        xLSTMCache()
    """

    def __init__(
        self,
        config: xLSTMConfig,
        max_batch_size: int,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[str] = None,
        **kwargs,
    ):
        self.seqlen_offset = 0
        self.dtype = dtype
        self.config = config
        self.rnn_state = {
            layer: (
                torch.zeros(
                    [max_batch_size, config.num_heads, config.qk_head_dim, config.v_head_dim],
                    dtype=dtype,
                    device=device,
                ),
                torch.zeros([max_batch_size, config.num_heads, config.qk_head_dim], dtype=dtype, device=device),
                torch.zeros([max_batch_size, config.num_heads, 1], dtype=dtype, device=device),
            )
            for layer in range(config.num_hidden_layers)
        }

    def reset(self):
        self.rnn_state = {
            layer: (
                torch.zeros_like(self.rnn_state[layer][0]),
                torch.zeros_like(self.rnn_state[layer][1]),
                torch.zeros_like(self.rnn_state[layer][2]),
            )
            for layer in self.rnn_state
        }


@dataclass
@auto_docstring
class xLSTMOutput(ModelOutput):
    r"""
    cache_params (`xLSTMCache`):
        The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
        avoid providing the old `input_ids`.
    """

    last_hidden_state: Optional[torch.FloatTensor]
    cache_params: Optional[xLSTMCache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None


@auto_docstring
class xLSTMModel(xLSTMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # use embbeding_dim and num_blocks once here to make use of them
        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.blocks = nn.ModuleList([xLSTMBlock(config) for _ in range(config.num_blocks)])
        self.out_norm = xLSTMRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embedding):
        self.embeddings = new_embedding

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[xLSTMCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, xLSTMOutput]:
        r"""
        cache_params (`xLSTMCache`, *optional*):
            The xLSTMCache that carries the RNN states.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if use_cache and cache_params is None:
            cache_params = xLSTMCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        hidden_states = inputs_embeds

        if (
            not self.training
            and self.config.max_inference_chunksize < hidden_states.shape[1]
            and not output_hidden_states
        ):
            offset = 0
            with torch.no_grad():
                if cache_params is None:
                    cache_params = xLSTMCache(config=self.config, max_batch_size=hidden_states.shape[0])
                final_state = torch.zeros_like(hidden_states)
                while offset < hidden_states.shape[1]:
                    hidden_states_chunk = hidden_states[
                        :, offset : min(offset + self.config.max_inference_chunksize, hidden_states.shape[1])
                    ]
                    for layer_idx, xlstm_block in enumerate(self.blocks):
                        hidden_states_chunk, rnn_state = xlstm_block(
                            hidden_states_chunk,
                            state=cache_params.rnn_state[layer_idx],
                        )
                        for state_idx in range(len(cache_params.rnn_state[layer_idx])):
                            local_rnn_state = rnn_state[state_idx]
                            cache_params.rnn_state[layer_idx][state_idx].copy_(local_rnn_state)
                        cache_params.rnn_state_initial = False
                    final_state[
                        :, offset : min(offset + self.config.max_inference_chunksize, hidden_states.shape[1])
                    ] = hidden_states_chunk
                    offset += self.config.max_inference_chunksize
                hidden_states = final_state
        else:
            all_hidden_states = () if output_hidden_states else None
            for layer_idx, xlstm_block in enumerate(self.blocks):
                if self.gradient_checkpointing and self.training:
                    hidden_states, rnn_state = self._gradient_checkpointing_func(
                        xlstm_block.__call__,
                        hidden_states,
                        cache_params.rnn_state[layer_idx] if cache_params is not None else None,
                    )
                else:
                    hidden_states, rnn_state = xlstm_block(
                        hidden_states,
                        state=cache_params.rnn_state[layer_idx] if cache_params is not None else None,
                    )
                if cache_params:
                    for state_idx in range(len(cache_params.rnn_state[layer_idx])):
                        local_rnn_state = rnn_state[state_idx]
                        cache_params.rnn_state[layer_idx][state_idx].copy_(local_rnn_state)
                    cache_params.rnn_state_initial = False

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.out_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return xLSTMOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params,
            hidden_states=all_hidden_states,
        )


@dataclass
@auto_docstring
class xLSTMCausalLMOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    cache_params (`xLSTMCache`, *optional*, carrying the RNN states):
        The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
        avoid providing the old `input_ids`.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[xLSTMCache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None


@auto_docstring
class xLSTMForCausalLM(xLSTMPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = xLSTMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,  # not used but needed, otherwise generate complains when passing tokenizer inputs
        inputs_embeds=None,
        use_cache=None,
        cache_params: Optional[xLSTMCache] = None,
        **kwargs,
    ):
        if use_cache and cache_params is not None:
            # If the first cache position is non-zero, we assume we are in generation mode.
            # Thus, the cache_params state is assumed to be the state before the last token
            # (lastly generated token), and all previous tokens are already ingested.
            # This should as well support generation from scratch with the [BOS] token inserted first.
            input_ids = input_ids[:, -1:]
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, -1:]

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({"cache_params": cache_params, "use_cache": use_cache})
        return model_inputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[xLSTMCache] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, xLSTMCausalLMOutput]:
        r"""
        cache_params (`xLSTMCache`, *optional*):
            The xLSTMCache that carries the RNN states.
        """
        xlstm_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        hidden_states = xlstm_outputs[0]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        if not self.training and self.config.max_inference_chunksize < logits.shape[1]:
            offset = 0
            with torch.no_grad():
                while offset < logits.shape[1]:
                    logits[:, offset : min(offset + self.config.max_inference_chunksize, logits.shape[1])] = soft_cap(
                        logits[:, offset : min(offset + self.config.max_inference_chunksize, logits.shape[1])],
                        self.config.output_logit_soft_cap,
                    )
                    offset += self.config.max_inference_chunksize
        else:
            logits = soft_cap(logits, self.config.output_logit_soft_cap)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < nstate predict nstate
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return xLSTMCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=xlstm_outputs.cache_params,
            hidden_states=xlstm_outputs.hidden_states,
        )


__all__ = [
    "xLSTMForCausalLM",
    "xLSTMModel",
    "xLSTMPreTrainedModel",
]
