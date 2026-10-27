# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Expert parallelism (EP) for MoE models: the EP styles and the plan application."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, NamedTuple

from ..utils import logging
from ..utils.generic import GeneralInterface
from ..utils.import_utils import is_torch_available, is_torch_distributed_available
from .parallel_utils import ALL_PARALLEL_STYLES, _same_ranks, _use_local_dtensor_params


if TYPE_CHECKING:
    from torch import nn
    from torch.distributed.device_mesh import DeviceMesh


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

if is_torch_distributed_available():
    from torch.distributed import ReduceOp, all_gather_into_tensor, all_reduce, all_to_all_single, get_backend
    from torch.distributed._functional_collectives import all_to_all_single_autograd
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

    from .parallel_utils import _all_reduce_backward, _AllReduceForward, _WaitTensor


class ExpertParallelLayer:
    """Base class of the expert-parallel styles, with the same hooks as `TensorParallelLayer`."""

    def should_use_local_tensors(self, module):
        """Whether this module's forward requires local inputs and parameters."""
        return False

    def validate_param(self, module, param, mesh, parameter_name=None):
        """Validate a parameter before applying this EP style."""
        pass

    def shard_param(self, module, param, mesh):
        """Wrap ONE parameter as a DTensor placeholder. Default: no-op."""
        pass

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        return args, kwargs

    def context_around_forward(self, module, mesh):
        if self.should_use_local_tensors(module):
            return _use_local_dtensor_params(module)
        return contextlib.nullcontext()

    def transform_output_post_forward(self, module, output, mesh):
        return output

    def install_forward(self, module, mesh):
        """Install pre / around / post transforms by replacing module.forward."""
        original_forward = module.forward

        def ep_forward(*args, **kwargs):
            args, kwargs = self.transform_inputs_pre_forward(module, args, kwargs, mesh)
            with self.context_around_forward(module, mesh):
                output = original_forward(*args, **kwargs)
            return self.transform_output_post_forward(module, output, mesh)

        module.forward = ep_forward
        return module


class MoEParamShard(ExpertParallelLayer):
    """Param-only EP style for MoE expert weights (``grouped_gemm``).

    Shards dim 0 and updates module.num_experts to the per-rank local count so the
    experts forward and ep_router sentinel agree.
    """

    def __init__(self, placement, *, shards_expert_dim: bool = False):
        self.placement = placement
        self.shards_expert_dim = shards_expert_dim

    def shard_param(self, module, param, mesh):
        meta = module._parameters.get(param)
        if meta is None:
            return
        if self.shards_expert_dim and hasattr(module, "num_experts"):
            global_num_experts = meta.shape[0]
            expert_parallel_size = mesh.size()
            if global_num_experts % expert_parallel_size != 0:
                raise ValueError(
                    f"Cannot evenly shard {global_num_experts} experts across "
                    f"{expert_parallel_size} expert-parallel ranks."
                )
            module.num_experts = global_num_experts // expert_parallel_size
            # The experts forward masks sentinel rows only when its experts are actually split.
            module._is_expert_parallel = True
        module._parameters[param] = torch.nn.Parameter(
            distribute_tensor(meta, mesh, [self.placement], src_data_rank=None),
            requires_grad=meta.requires_grad,
        )


class EpRouterParallel(ExpertParallelLayer):
    """Expert-parallel router: forward-only slicing of router outputs to local experts.

    Expects the router to return `(router_logits, router_scores, router_indices, *extra)`.
    `router_logits` and any trailing `extra` outputs are passed through unchanged.

    The gate runs replicated on every rank and emits global expert IDs and scores.
    Under EP each rank owns `num_experts // ep_size` experts, so this post-forward hook:

    - zeroes scores for non-local experts
    - remaps surviving global indices to local indices (`fmod` after masking non-local slots)
    - sets dropped slots to sentinel `num_local_experts` (skipped by grouped_gemm experts forward)

    Downstream `moe_tp_experts` allreduce-sums partial per-rank expert outputs.

    Example: 4 tokens, top_k=4, 128 experts, EP=8 → num_local_experts=16.

    Router output (identical on all ranks):
        router_indices:
            [ 52,  42, 119,  67],
            [102,  89,  61,  40],
            [ 82, 103,   4,  34],
            [ 93,  23, 109,  11]

    Owning rank (`index // 16`):
            [ 3,  2,  7,  4],
            [ 6,  5,  3,  2],
            [ 5,  6,  0,  2],
            [ 5,  1,  6,  0],

    After slicing on rank 0 (owns experts 0-15):
        router_indices:          router_scores (illustrative):
            [16, 16, 16, 16],       [0.0, 0.0, 0.0, 0.0],
            [16, 16, 16, 16],       [0.0, 0.0, 0.0, 0.0],
            [16, 16,  4, 16],       [0.0, 0.0, 0.3, 0.0],
            [16, 16, 16, 11],       [0.0, 0.0, 0.0, 0.1],

    On rank 1 (owns experts 16-31), global expert 23 remaps to local index 7 via `23 % 16`.

    Scores and indices stay paired element-wise in `(seq, top_k)` shape.
    """

    def transform_output_post_forward(self, module, output, mesh):
        ep_rank, ep_size = mesh.get_local_rank(), mesh.size()
        num_experts = getattr(module, "num_experts", None)
        if num_experts is None:
            num_experts = getattr(getattr(module, "config", None), "num_experts", None)
        if num_experts is None:
            raise AttributeError(
                f"Router module {type(module).__name__} is missing `num_experts` and `config.num_experts`"
            )
        if num_experts % ep_size != 0:
            raise ValueError(f"num_experts must be divisible by ep_size: {num_experts} % {ep_size} != 0")
        num_local_experts = num_experts // ep_size

        router_logits, router_scores, router_indices, *extra_outputs = output
        # Each rank's score gradient covers only its local experts' slots; sum the per-rank partials
        # before the mask (each slot has exactly one owning rank, so the sum is exact).
        process_group = mesh.get_group() if mesh.ndim == 1 else mesh.get_group("tp")
        router_scores = _all_reduce_backward(router_scores, process_group)
        non_local_mask = (router_indices // num_local_experts) != ep_rank
        router_scores = router_scores.masked_fill(non_local_mask, 0.0)
        router_indices = router_indices.masked_fill(non_local_mask, -1)
        if num_local_experts > 1:
            router_indices = torch.fmod(router_indices, num_local_experts)
        else:
            router_indices = router_indices.masked_fill(router_indices > 0, 0).masked_fill(router_indices < 0, -1)
        router_indices = router_indices.masked_fill(router_indices == -1, num_local_experts)
        return router_logits, router_scores, router_indices, *extra_outputs


def _slice_over_tp(tp_mesh, *tensors):
    """Each rank of a TP group keeps a disjoint slice of the batch the group shares (`AllToAllEP` shows why)."""
    if tp_mesh is None or tp_mesh.size() == 1:
        return tensors
    # Replicate -> Shard(0) is a local chunk (no communication); its backward all-gathers the row gradients.
    return tuple(
        DTensor.from_local(t, tp_mesh, [Replicate()], run_check=False).redistribute(tp_mesh, [Shard(0)]).to_local()
        for t in tensors
    )


def _gather_over_tp(tp_mesh, output, num_tokens):
    """Gathers the slices `_slice_over_tp` kept back into the batch of `num_tokens` the TP group shares."""
    if tp_mesh is None or tp_mesh.size() == 1:
        return output
    # Shard(0) -> Replicate is one all-gather of the row slices, uneven or empty ones too.
    hidden_dim = output.size(-1)
    output = DTensor.from_local(
        output.contiguous(), tp_mesh, [Shard(0)], shape=(num_tokens, hidden_dim), stride=(hidden_dim, 1)
    )
    return output.full_tensor()


class EPImplementation:
    """
    How the experts of an `ep_dispatch_experts` module get their tokens. One instance is built per experts module when
    the plan is applied, so an implementation can keep state across forwards (communication buffers, a layout it
    reuses).

    A forward is `dispatch` -> the local experts -> `combine`:

    - `dispatch(hidden_states, top_k_index, top_k_weights)` takes the batch this rank holds (global expert ids) and
      returns the rows its local experts run on, their local expert ids and weights, and the state `combine` needs.
    - `combine(expert_output, state)` returns the weighted expert output for each token of the batch this rank holds.

    Gradients must reach every rank that holds a token, routing weights included, so a collective that autograd does not
    differentiate has to be wrapped in a `torch.autograd.Function`.
    """

    # Set by implementations that only run when every rank of the EP group holds the same batch, e.g. `AllReduceEP`.
    needs_shared_batch = False

    def __init__(self, experts, ep_mesh, tp_mesh=None):
        self.experts = experts
        self.experts_forward = experts.forward  # the experts' own forward, on local expert ids
        self.num_local_experts = experts.num_experts
        self.ep_mesh = ep_mesh
        self.tp_mesh = tp_mesh
        self.shares_batch = tp_mesh is not None and _same_ranks(tp_mesh, ep_mesh)
        if self.needs_shared_batch and not self.shares_batch:
            raise ValueError(
                f"{type(self).__name__} needs every rank of the expert group to hold the same batch (`ep_size == tp_size`)."
            )

    def dispatch(self, hidden_states, top_k_index, top_k_weights):
        raise NotImplementedError

    def combine(self, expert_output, state):
        raise NotImplementedError

    def __call__(self, hidden_states, top_k_index, top_k_weights):
        with _use_local_dtensor_params(self.experts):
            tokens, expert_ids, weights, state = self.dispatch(hidden_states, top_k_index, top_k_weights)
            if tokens.shape[0] == 0:
                # Keep the empty rows connected to backward, so the reverse exchange and FSDP's reduction still run.
                expert_output = self.experts_forward(
                    tokens.sum(dim=0, keepdim=True), expert_ids.new_zeros((1, 1)), weights.new_ones((1, 1))
                )[:0]
            else:
                expert_output = self.experts_forward(tokens, expert_ids, weights)
            output = self.combine(expert_output, state)
        return output.to(hidden_states.dtype)


class _AllToAllState(NamedTuple):
    top_k_weights: torch.Tensor
    order: torch.Tensor
    send_sizes: list[int]
    recv_sizes: list[int]
    num_tokens: int


class AllToAllEP(EPImplementation):
    """
    The ranks of a TP group keep disjoint slices of the batch they share, send each (token, expert) pair to the
    expert's rank with an all-to-all and get the results back with a second one, then gather the slices back over TP.
    The split sizes are read back to the host every layer, so it does not capture in a CUDA graph.

    Example: 8 experts [E0, E7] with DistributedConfig(tp_size=2, fsdp_size=4, ep_size=4):
        - fsdp_size=4: 4 batches [B0, B3]
        - tp_size=2: the 2 ranks of a TP group share one batch
        - efsdp_size = fsdp_size * tp_size / ep_size = 4 * 2 / 4 = 2 copies of each expert

    GPU         0      1       2      3       4      5       6      7
                |      |       |      |       |      |       |      |
    dense view  ---------------------------------------------------------
    batch       [====B0====]   [====B1====]   [====B2====]   [====B3====]
    tp_size     [___________ 0 ___________]   [___________ 1 ___________]
    fsdp_size        0              1              2              3

    expert view ---------------------------------------------------------

    experts     E0E1   E2E3    E4E5   E6E7    E0E1   E2E3    E4E5   E6E7
    ep_size      0      1       2      3       0      1       2      3
    efsdp_size [___________ 0 ___________]   [___________ 1 ___________]

    The TP slicing keeps a token from being computed twice. Say token 1 of B0 picks E4, which lives on rank 2. B0 sits
    on ranks 0 and 1, so if both sent it, E4 would compute it twice: token 1 falls in rank 0's slice, so rank 0 sends it
    and rank 1 does not.
    """

    def dispatch(self, hidden_states, top_k_index, top_k_weights):
        num_tokens = hidden_states.size(0)
        hidden_states, top_k_index, top_k_weights = _slice_over_tp(
            self.tp_mesh, hidden_states, top_k_index, top_k_weights
        )
        ep_group, ep_size = self.ep_mesh.get_group(), self.ep_mesh.size()
        num_top_k = top_k_index.size(-1)

        # Sorting the selected pairs by expert groups them by owner rank, since each rank owns a contiguous range of
        # experts, and the per-expert counts tell every receiver which expert each token it gets is for. The split
        # sizes are the one host sync of the layer.
        expert_ids = top_k_index.reshape(-1)
        order = torch.argsort(expert_ids)
        send_tokens = hidden_states[order // num_top_k]
        send_counts = torch.zeros(self.num_local_experts * ep_size, dtype=torch.long, device=hidden_states.device)
        send_counts = send_counts.scatter_add_(0, expert_ids, torch.ones_like(expert_ids))
        send_counts = send_counts.view(ep_size, self.num_local_experts)
        recv_counts = torch.empty_like(send_counts)
        all_to_all_single(recv_counts, send_counts, group=ep_group)
        send_sizes, recv_sizes = torch.stack([send_counts.sum(dim=1), recv_counts.sum(dim=1)]).tolist()
        recv_tokens = all_to_all_single_autograd(send_tokens, recv_sizes, send_sizes, ep_group)
        # hand the experts the received rows, not the all-to-all's pending result (whose pointer is null until
        # waited on): https://github.com/huggingface/transformers/pull/49421
        recv_tokens = _WaitTensor.apply(recv_tokens)
        recv_expert_ids = torch.arange(self.num_local_experts, device=hidden_states.device).repeat(ep_size)
        recv_expert_ids = recv_expert_ids.repeat_interleave(recv_counts.reshape(-1), output_size=sum(recv_sizes))
        recv_expert_ids = recv_expert_ids.unsqueeze(-1)
        # Experts run top-1 with unit weights; the routing weights are applied in `combine`, on the sender.
        recv_weights = torch.ones_like(recv_expert_ids, dtype=recv_tokens.dtype)
        state = _AllToAllState(top_k_weights, order, send_sizes, recv_sizes, num_tokens)
        return recv_tokens, recv_expert_ids, recv_weights, state

    def combine(self, expert_output, state):
        hidden_dim = expert_output.size(-1)
        recv_out = all_to_all_single_autograd(
            expert_output, state.send_sizes, state.recv_sizes, self.ep_mesh.get_group()
        )
        # Restore the original (token, top-k slot) order, then apply routing weights.
        token_outputs = torch.empty_like(recv_out)
        token_outputs[state.order] = recv_out
        token_outputs = token_outputs.view(*state.top_k_weights.shape, hidden_dim)
        output = (token_outputs * state.top_k_weights.unsqueeze(-1)).sum(dim=1)
        return _gather_over_tp(self.tp_mesh, output, state.num_tokens)


class AllReduceEP(EPImplementation):
    """
    When the EP group shares one batch (`ep_size == tp_size`): every rank runs its own experts on the whole batch,
    masking the routes to other ranks' experts (local id `num_local_experts`, weight 0), and an all-reduce sums the
    partial outputs. No tokens move and no split size is read back, so it captures in a CUDA graph; the masked routes
    still go through the experts' gather and elementwise passes, so its cost grows with the batch.

    Example: 4 experts on 2 ranks, top-2. Token t picks E1 (weight w1) and E2 (weight w2).

    GPU                    0                      1
                           |                      |
    batch           [====== B ======]      [====== B ======]    the same tokens on both ranks
    experts             E0    E1               E2    E3
    top_k_index[t]       [1, 2]                 [1, 2]          global expert ids
    local index[t]       [1, 2]                 [2, 0]          2 = not on this rank, masked
    top_k_weights[t]     [w1, 0]                [0, w2]
    expert output[t]    w1 * E1(t)             w2 * E2(t)
    after all-reduce          w1 * E1(t) + w2 * E2(t) on both ranks
    """

    needs_shared_batch = True

    def dispatch(self, hidden_states, top_k_index, top_k_weights):
        ep_group = self.ep_mesh.get_group()
        # Each rank backpropagates through its own experts only, so the gradients of the shared inputs it computes are
        # partial: sum them over the group (the forward all-reduce has an identity backward).
        hidden_states = _all_reduce_backward(hidden_states, ep_group)
        top_k_weights = _all_reduce_backward(top_k_weights, ep_group)
        off_rank = (top_k_index // self.num_local_experts) != self.ep_mesh.get_local_rank()
        local_index = torch.where(off_rank, self.num_local_experts, top_k_index % self.num_local_experts)
        return hidden_states, local_index, top_k_weights.masked_fill(off_rank, 0.0), None

    def combine(self, expert_output, state):
        return _AllReduceForward.apply(expert_output, self.ep_mesh.get_group())


class DeepEP(EPImplementation):
    """
    DeepEP's dispatch and combine (`kernels-community/deep-ep`), in its expanded layout: each rank receives one row per
    (token, local expert) slot, grouped by local expert, with that slot's routing weight, so its experts only see their
    own routes and apply the weights, and the combine sums each token's rows back on the token's rank. The ranks of a
    TP group each send a disjoint slice of the batch they share, as with `AllToAllEP`. Tokens travel in bfloat16.

    Eagerly, DeepEP reads the received counts back to the host, so the experts get exactly the received rows. Under CUDA
    graph capture or `torch.compile`, it runs without that sync: the received rows are sized for the worst case (every
    rank sending its whole capacity of tokens with all of their top-k experts here), so they cannot overflow, and the
    rows past the received total are zeroed and skipped by the experts. The capacity is the batch per rank rounded up to
    a power of two, and must be the same on every rank of the EP group: eagerly, ranks holding different batches agree
    on it with one small all-reduce; under capture or compilation, each rank derives it from its own batch, so the
    ranks of a group must run batches of the same size (as padded CUDA graph batches are).

    Every layer of an EP group shares one DeepEP buffer, built on the first forward and rebuilt larger when a batch
    outgrows it. Buffers are never allocated during capture: run the batch size eagerly once before capturing it (the
    usual warm-up). `transformers.integrations.deepep.destroy_deepep_buffers()` frees them; call it on every rank
    before destroying the process groups.

    Needs a GPU of compute capability 9.0 or newer, the NCCL PyTorch loads ≥ 2.32.3 and a CUDA toolkit ≥ 13.1 (`nvcc`)
    at runtime; building the implementation raises an `ImportError` naming the missing requirement.
    """

    def __init__(self, experts, ep_mesh, tp_mesh=None):
        super().__init__(experts, ep_mesh, tp_mesh)
        from ..integrations.deepep import load_deepep_kernel

        load_deepep_kernel()

    def tokens_per_rank(self, num_tokens, sync_free, device):
        """The most tokens a rank of the EP group sends: the largest TP slice. Eagerly, ranks holding different
        batches agree on it with one small all-reduce; under capture or compilation, the ranks must run batches of the
        same size."""
        tp_size = self.tp_mesh.size() if self.tp_mesh is not None else 1
        tokens = max(1, -(-num_tokens // tp_size))
        if sync_free or self.shares_batch:
            return tokens
        tokens = torch.tensor([tokens], device=device)
        all_reduce(tokens, op=ReduceOp.MAX, group=self.ep_mesh.get_group())
        return int(tokens.item())

    def dispatch(self, hidden_states, top_k_index, top_k_weights):
        from ..integrations.deepep import deepep_dispatch

        num_tokens = hidden_states.size(0)
        sync_free = torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()
        capacity = _power_of_two_at_least(self.tokens_per_rank(num_tokens, sync_free, hidden_states.device))
        hidden_states, top_k_index, top_k_weights = _slice_over_tp(
            self.tp_mesh, hidden_states, top_k_index, top_k_weights
        )
        tokens, expert_ids, weights, handle = deepep_dispatch(
            hidden_states,
            top_k_index,
            top_k_weights,
            self.ep_mesh.get_group(),
            self.num_local_experts * self.ep_mesh.size(),
            capacity,
            cpu_sync=not sync_free,
        )
        # one slot per row: the experts run top-1 and apply the slot's routing weight, the combine only sums
        tokens = tokens.to(hidden_states.dtype)
        return tokens, expert_ids.unsqueeze(-1), weights.unsqueeze(-1).to(tokens.dtype), (handle, num_tokens)

    def combine(self, expert_output, state):
        from ..integrations.deepep import deepep_combine

        handle, num_tokens = state
        return _gather_over_tp(self.tp_mesh, deepep_combine(expert_output, handle), num_tokens)


class AutoEP(EPImplementation):
    """
    The default: `AllReduceEP` at inference (gradients disabled) when the EP group shares one batch, and token
    dispatch in training and wherever the ranks hold different batches. Token dispatch runs `DeepEP` where DeepEP
    runs (see `is_deepep_loadable`), and `AllToAllEP` otherwise. At inference, the all-reduce and DeepEP never read
    split sizes back to the host under capture, so decode captures in a CUDA graph.
    """

    def __init__(self, experts, ep_mesh, tp_mesh=None):
        super().__init__(experts, ep_mesh, tp_mesh)
        self.all_to_all = AllToAllEP(experts, ep_mesh, tp_mesh)
        self.deepep = DeepEP(experts, ep_mesh, tp_mesh) if _deepep_runs(ep_mesh) else None
        # Only built where it can run: an EP group whose ranks hold different batches has no all-reduce path.
        self.all_reduce = AllReduceEP(experts, ep_mesh, tp_mesh) if self.shares_batch else None

    def __call__(self, hidden_states, top_k_index, top_k_weights):
        if self.all_reduce is not None and not torch.is_grad_enabled():
            return self.all_reduce(hidden_states, top_k_index, top_k_weights)
        if self.deepep is None:
            return self.all_to_all(hidden_states, top_k_index, top_k_weights)
        return self.deepep(hidden_states, top_k_index, top_k_weights)


def _power_of_two_at_least(n):
    """DeepEP sizes its exchange by tokens per rank; rounding them up to a power of two lets batch sizes share it."""
    capacity = 1
    while capacity < n:
        capacity *= 2
    return capacity


def _deepep_runs(ep_mesh) -> bool:
    """Whether `AutoEP` dispatches with DeepEP: an NCCL EP group and DeepEP loadable."""
    if get_backend(ep_mesh.get_group()) != "nccl":
        return False
    from ..integrations.deepep import is_deepep_loadable

    if is_deepep_loadable():
        return True
    try:
        is_deepep_loadable(raise_error=True)
    except ImportError as error:
        logger.info_once(f'`ep_implementation="auto"` dispatches tokens with all-to-all, not DeepEP: {error}')
    return False


class PassthroughEP(EPImplementation):
    """Pass through a batch to an expert kernel that performs its own dispatch and combine."""

    needs_shared_batch = True

    def __call__(self, hidden_states, top_k_index, top_k_weights):
        with _use_local_dtensor_params(self.experts):
            output = self.experts_forward(hidden_states, top_k_index, top_k_weights, self.ep_mesh.get_group())
        return output.to(hidden_states.dtype)


class MegaMoEEP(EPImplementation):
    """
    DeepGEMM Mega MoE, whose kernel fuses the token dispatch, the experts and the combine. The kernel treats each
    rank's rows as that rank's own tokens, while every rank of the EP group holds the same batch here: each rank hands
    it an equal slice of that batch, with global expert ids and unmasked weights, and the combined slices are gathered
    back, so every token goes through the experts once. The kernel exchanges tokens through symmetric-memory buffers,
    which this implementation keeps per EP group and shares between every layer, since the layers run one after another.
    The kernel has no backward, so this runs at inference only.
    """

    # TODO: Mega MoE is built for ranks holding different batches (data-parallel attention + EP), where neither the
    # slicing nor the gather is needed; until inference supports data parallelism, it runs on the shared batch.
    needs_shared_batch = True
    # EP group name -> the kernel's symmetric buffers, by shape; `symm_buffers.clear()` frees them
    symm_buffers: dict[str, dict] = {}

    def __call__(self, hidden_states, top_k_index, top_k_weights):
        ep_group, ep_size, ep_rank = self.ep_mesh.get_group(), self.ep_mesh.size(), self.ep_mesh.get_local_rank()
        num_tokens = hidden_states.size(0)
        # equal slices, the last one padded with skipped slots (expert id -1, weight 0), so the outputs gather evenly
        rows_per_rank = max(1, -(-num_tokens // ep_size))
        padding = rows_per_rank * ep_size - num_tokens
        if padding:
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, padding))
            top_k_index = torch.nn.functional.pad(top_k_index, (0, 0, 0, padding), value=-1)
            top_k_weights = torch.nn.functional.pad(top_k_weights, (0, 0, 0, padding))
        rows = slice(ep_rank * rows_per_rank, (ep_rank + 1) * rows_per_rank)

        symm_buffers = self.symm_buffers.setdefault(ep_group.group_name, {})
        with _use_local_dtensor_params(self.experts):
            local_output = self.experts_forward(
                hidden_states[rows], top_k_index[rows], top_k_weights[rows], ep_group, symm_buffers
            )
        output = local_output.new_empty(rows_per_rank * ep_size, local_output.size(-1))
        all_gather_into_tensor(output, local_output.contiguous(), group=ep_group)
        return output[:num_tokens].to(hidden_states.dtype)


class EPImplementationInterface(GeneralInterface):
    """The `EPImplementation` classes `DistributedConfig.ep_implementation` can name. Register one under a new name to add one."""

    _global_mapping = {
        "auto": AutoEP,
        "all_to_all": AllToAllEP,
        "all_reduce": AllReduceEP,
        "deepep": DeepEP,
        "passthrough": PassthroughEP,
        "deepgemm_megamoe": MegaMoEEP,
    }


ALL_EP_IMPLEMENTATIONS: EPImplementationInterface = EPImplementationInterface()


class EpDispatchExpertsParallel(ExpertParallelLayer):
    """Whole experts per rank (`Shard(0)` on the EP mesh), reached through an `EPImplementation` from
    `ALL_EP_IMPLEMENTATIONS`, picked by `DistributedConfig.ep_implementation`."""

    def install_forward(self, module, ep_mesh, *, tp_mesh=None, ep_implementation="auto"):
        """`tp_mesh` is the group of ranks holding the same batch."""
        if ep_implementation not in ALL_EP_IMPLEMENTATIONS:
            raise ValueError(
                f"`ep_implementation={ep_implementation!r}` names no registered implementation: use one of "
                f"{', '.join(repr(name) for name in ALL_EP_IMPLEMENTATIONS)}, or register an `EPImplementation` in "
                "`ALL_EP_IMPLEMENTATIONS`."
            )
        implementation = ALL_EP_IMPLEMENTATIONS[ep_implementation](module, ep_mesh, tp_mesh)

        def ep_forward(hidden_states, top_k_index, top_k_weights):
            if isinstance(hidden_states, DTensor):
                hidden_states = hidden_states.to_local()
            if isinstance(top_k_weights, DTensor):
                top_k_weights = top_k_weights.to_local()
            return implementation(hidden_states, top_k_index, top_k_weights)

        module.forward = ep_forward
        return module


if is_torch_distributed_available():
    ALL_PARALLEL_STYLES.register("grouped_gemm", MoEParamShard(Shard(0), shards_expert_dim=True))
    ALL_PARALLEL_STYLES.register("ep_router", EpRouterParallel())
    ALL_PARALLEL_STYLES.register("ep_dispatch_experts", EpDispatchExpertsParallel())


def apply_expert_parallelism(
    model: nn.Module, ep_mesh: DeviceMesh, tp_mesh: DeviceMesh, plan: dict[str, str], ep_implementation: str = "auto"
):
    """
    Shard experts on EP; `ep_dispatch_experts` modules reach them through the `ep_implementation` registered in
    `ALL_EP_IMPLEMENTATIONS`.
    """
    modules = dict(model.named_modules())
    parameters = dict(model.named_parameters())

    for name, style_name in plan.items():
        style = ALL_PARALLEL_STYLES[style_name]
        if name in parameters:
            module_name, _, param_name = name.rpartition(".")
            module = modules[module_name]
            style.validate_param(module, param_name, ep_mesh, parameter_name=name)
            style.shard_param(module, param_name, ep_mesh)
        else:
            module = modules[name]
            # Dispatch hooks need both meshes to redistribute tokens between TP and EP ranks.
            if style_name == "ep_dispatch_experts":
                style.install_forward(module, ep_mesh, tp_mesh=tp_mesh, ep_implementation=ep_implementation)
            else:
                style.install_forward(module, ep_mesh)
        module._is_hooked = True

    return model
