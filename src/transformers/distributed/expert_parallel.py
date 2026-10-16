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
"""How the experts of an `ep_dispatch_experts` module get their tokens: the `EpStrategy` classes `ep_implementation` names."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils.generic import GeneralInterface
from ..utils.import_utils import is_torch_available, is_torch_distributed_available
from ._ops import _AllReduceBackward, _AllReduceForward, _use_local_dtensor_params
from .tensor_parallel import ALL_PARALLEL_STYLES, TensorParallelLayer, _get_parameter_plan


if TYPE_CHECKING:
    from torch import nn
    from torch.distributed.device_mesh import DeviceMesh


if is_torch_available():
    import torch

if is_torch_distributed_available():
    import torch.distributed as dist
    from torch.distributed._functional_collectives import all_to_all_single, wait_tensor
    from torch.distributed.tensor import DTensor, Partial, Replicate, Shard


class EpStrategy:
    """
    How an `ep_dispatch_experts` module's experts get their tokens. One instance is built per experts module when the
    plan is applied, so a strategy can keep state across forwards (communication buffers, a layout it reuses).

    A forward is `dispatch` -> the local experts -> `combine`:

    - `dispatch(hidden_states, top_k_index, top_k_weights)` takes the batch this rank holds (global expert ids) and
      returns the rows its local experts run on, their local expert ids and weights, and whatever `combine` needs.
    - `combine(expert_output, state)` returns the weighted expert output for each token of the batch this rank holds.

    Gradients must reach every rank that holds a token, routing weights included: wrap collectives that autograd does
    not differentiate (DeepEP's dispatch and combine) in a `torch.autograd.Function`. For example:

        class DeepEPStrategy(EpStrategy):
            def __init__(self, experts, ep_mesh, *, tp_mesh=None, distributed_config=None):
                super().__init__(experts, ep_mesh, tp_mesh=tp_mesh, distributed_config=distributed_config)
                self.buffer = get_buffer(ep_mesh.get_group())  # one DeepEP buffer per group, shared by the layers

            def dispatch(self, hidden_states, top_k_index, top_k_weights):
                ...  # DeepEP dispatch: received rows, their local expert ids and weights, the DeepEP handle

            def combine(self, expert_output, handle):
                ...  # DeepEP combine with the handle

        ALL_EP_STRATEGIES.register("deepep", DeepEPStrategy)
        DistributedConfig(tp_size=1, fsdp_size=8, ep_size=8, ep_implementation="deepep")
    """

    # Only runs when every rank of the EP group holds the same batch (`ep_size == tp_size`).
    needs_shared_batch = False

    def __init__(self, experts, ep_mesh, *, tp_mesh=None, distributed_config=None):
        self.experts = experts
        self.experts_forward = experts.forward  # the experts' own forward, on local expert ids
        self.num_local_experts = experts.num_experts
        self.ep_mesh = ep_mesh
        self.tp_mesh = tp_mesh
        self.shares_batch = tp_mesh is not None and dist.get_process_group_ranks(
            tp_mesh.get_group()
        ) == dist.get_process_group_ranks(ep_mesh.get_group())
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


class AllToAllStrategy(EpStrategy):
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

    The EP group (4 ranks) spans 2 batches, so only all-to-all applies; with tp_size == ep_size, every rank of the EP
    group would hold the same batch and `AllReduceStrategy` would apply too.

    The TP slicing is what keeps a token from being computed twice. Say token 1 of B0 picks E4, which lives on rank 2. B0 sits on ranks 0 and 1, so if both sent it, E4 would compute it
    twice: token 1 falls in rank 0's slice, so rank 0 sends it and rank 1 does not.
    """

    def dispatch(self, hidden_states, top_k_index, top_k_weights):
        num_tokens = hidden_states.size(0)
        if self.tp_mesh is not None and self.tp_mesh.size() > 1:
            # Replicate -> Shard(0) is a local chunk (no communication); its backward all-gathers the row gradients.
            hidden_states, top_k_index, top_k_weights = (
                DTensor.from_local(t, self.tp_mesh, [Replicate()], run_check=False)
                .redistribute(self.tp_mesh, [Shard(0)])
                .to_local()
                for t in (hidden_states, top_k_index, top_k_weights)
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
        torch.distributed.all_to_all_single(recv_counts, send_counts, group=ep_group)
        send_sizes, recv_sizes = torch.stack([send_counts.sum(dim=1), recv_counts.sum(dim=1)]).tolist()
        recv_tokens = all_to_all_single(send_tokens, recv_sizes, send_sizes, ep_group)
        # hand the experts the received rows, not the all-to-all's pending result (whose pointer is null until
        # waited on): https://github.com/huggingface/transformers/pull/49421
        recv_tokens = wait_tensor(recv_tokens)
        recv_expert_ids = torch.arange(self.num_local_experts, device=hidden_states.device).repeat(ep_size)
        recv_expert_ids = recv_expert_ids.repeat_interleave(recv_counts.reshape(-1), output_size=sum(recv_sizes))
        recv_expert_ids = recv_expert_ids.unsqueeze(-1)
        # Experts run top-1 with unit weights; the routing weights are applied in `combine`, on the sender.
        unit_weights = torch.ones_like(recv_expert_ids, dtype=recv_tokens.dtype)
        return recv_tokens, recv_expert_ids, unit_weights, (top_k_weights, order, send_sizes, recv_sizes, num_tokens)

    def combine(self, expert_output, state):
        top_k_weights, order, send_sizes, recv_sizes, num_tokens = state
        hidden_dim = expert_output.size(-1)
        recv_out = all_to_all_single(expert_output, send_sizes, recv_sizes, self.ep_mesh.get_group())
        # Restore the original (token, top-k slot) order, then apply routing weights.
        token_outputs = torch.empty_like(recv_out)
        token_outputs[order] = recv_out
        token_outputs = token_outputs.view(*top_k_weights.shape, hidden_dim)
        output = (token_outputs * top_k_weights.unsqueeze(-1)).sum(dim=1)
        if self.tp_mesh is None or self.tp_mesh.size() == 1:
            return output
        # Shard(0) -> Replicate is one all-gather of the row slices, uneven or empty ones too.
        output = DTensor.from_local(
            output.contiguous(), self.tp_mesh, [Shard(0)], shape=(num_tokens, hidden_dim), stride=(hidden_dim, 1)
        )
        return output.full_tensor()


class AllReduceStrategy(EpStrategy):
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
        # every rank backpropagates through its own experts only: sum the replicated inputs' gradients
        hidden_states = _AllReduceBackward.apply(hidden_states, ep_group)
        if torch.is_grad_enabled() and top_k_weights.requires_grad:
            top_k_weights = _AllReduceBackward.apply(top_k_weights, ep_group)
        off_rank = (top_k_index // self.num_local_experts) != self.ep_mesh.get_local_rank()
        local_index = torch.where(off_rank, self.num_local_experts, top_k_index % self.num_local_experts)
        return hidden_states, local_index, top_k_weights.masked_fill(off_rank, 0.0), None

    def combine(self, expert_output, state):
        return _AllReduceForward.apply(expert_output, self.ep_mesh.get_group())


class AutoStrategy(EpStrategy):
    """
    The default: `AllReduceStrategy` at inference (gradients disabled) when the EP group shares one batch, and
    `AllToAllStrategy` in training and wherever the ranks hold different batches. At inference, the all-reduce never reads
    split sizes back to the host, so decode captures in a CUDA graph.
    """

    def __init__(self, experts, ep_mesh, *, tp_mesh=None, distributed_config=None):
        super().__init__(experts, ep_mesh, tp_mesh=tp_mesh, distributed_config=distributed_config)
        kwargs = {"tp_mesh": tp_mesh, "distributed_config": distributed_config}
        self.all_to_all = AllToAllStrategy(experts, ep_mesh, **kwargs)
        self.all_reduce = AllReduceStrategy(experts, ep_mesh, **kwargs) if self.shares_batch else None

    def __call__(self, hidden_states, top_k_index, top_k_weights):
        if self.all_reduce is not None and not torch.is_grad_enabled():
            return self.all_reduce(hidden_states, top_k_index, top_k_weights)
        return self.all_to_all(hidden_states, top_k_index, top_k_weights)


class MegaMoEStrategy(EpStrategy):
    """
    DeepGEMM Mega MoE, whose kernel fuses the token dispatch, the experts and the combine. The kernel treats each
    rank's rows as that rank's own tokens, while every rank of the EP group holds the same batch here: each rank hands
    it an equal slice of that batch, with global expert ids and unmasked weights, and the combined slices are gathered
    back, so every token goes through the experts once. The kernel exchanges tokens through symmetric-memory buffers,
    which this strategy keeps per EP group and shares between every layer, since the layers run one after another.
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
        dist.all_gather_into_tensor(output, local_output.contiguous(), group=ep_group)
        return output[:num_tokens].to(hidden_states.dtype)


class EpStrategyInterface(GeneralInterface):
    """The `EpStrategy` classes `DistributedConfig.ep_implementation` can name. Register one under a new name to add one."""

    _global_mapping = {
        "auto": AutoStrategy,
        "all_to_all": AllToAllStrategy,
        "all_reduce": AllReduceStrategy,
        "deepgemm_megamoe": MegaMoEStrategy,
    }


ALL_EP_STRATEGIES: EpStrategyInterface = EpStrategyInterface()


def _resolve_ep_strategy(distributed_config) -> str:
    """The `ALL_EP_STRATEGIES` name an `ep_dispatch_experts` module runs: `ep_implementation`, `"auto"` when unset. A strategy
    that needs a shared batch (`"all_reduce"`) requires `ep_size == tp_size`."""
    ep_implementation = distributed_config.ep_implementation or "auto"
    if ep_implementation not in ALL_EP_STRATEGIES:
        raise ValueError(
            f"`ep_implementation={ep_implementation!r}` names no registered strategy: use one of "
            f"{', '.join(repr(name) for name in ALL_EP_STRATEGIES)}, or register an `EpStrategy` in `ALL_EP_STRATEGIES`."
        )
    if (
        ALL_EP_STRATEGIES[ep_implementation].needs_shared_batch
        and distributed_config.ep_size != distributed_config.tp_size
    ):
        raise ValueError(
            f"`ep_implementation={ep_implementation!r}` requires `ep_size == tp_size`: every rank of the expert group must hold the "
            "same batch."
        )
    return ep_implementation


class MoeExpertsParallel(TensorParallelLayer):
    def should_use_local_tensors(self, module):
        return True

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh, *, is_expert_parallel=False):
        hidden_states, *routing_args = args
        tp_group = mesh.get_group() if mesh.ndim == 1 else mesh.get_group("tp")
        if isinstance(hidden_states, DTensor):
            hidden_states = hidden_states.to_local()
        hidden_states = _AllReduceBackward.apply(hidden_states, tp_group)

        if len(routing_args) >= 2:
            top_k_index, top_k_weights, *extra_args = routing_args
            if isinstance(top_k_weights, DTensor):
                top_k_weights = top_k_weights.to_local()
            if not is_expert_parallel:
                top_k_weights = _AllReduceBackward.apply(top_k_weights, tp_group)
            routing_args = [top_k_index, top_k_weights, *extra_args]

        return (hidden_states, *routing_args), kwargs

    def install_forward(self, module, mesh, *, is_expert_parallel=False):
        """Install the transforms but pass `is_expert_parallel` in the forward call."""
        original_forward = module.forward
        output_source = (
            Partial()
            if any(
                isinstance(param, DTensor) and any(not placement.is_replicate() for placement in param.placements)
                for param in module.parameters()
            )
            else Replicate()
        )

        def tp_forward(*args, **kwargs):
            args, kwargs = self.transform_inputs_pre_forward(
                module, args, kwargs, mesh, is_expert_parallel=is_expert_parallel
            )
            with self.context_around_forward(module, mesh):
                output = original_forward(*args, **kwargs)
            return self.transform_output_post_forward(module, output, mesh, source=output_source)

        module.forward = tp_forward
        return module

    def transform_output_post_forward(self, module, output, mesh, source=None):
        if output is None:
            return None

        has_sharded_parameters = any(
            isinstance(param, DTensor) and any(not placement.is_replicate() for placement in param.placements)
            for param in module.parameters()
        )
        if not has_sharded_parameters:
            return output

        process_group = mesh.get_group() if mesh.ndim == 1 else mesh.get_group("tp")
        return _AllReduceForward.apply(output, process_group)


class MoeIdentityParallel(TensorParallelLayer):
    """
    Used in longcat_flash zero_experts (nn.Identity) which return the same value on every GPU, but the
    moe_tp_experts will sums across GPUs. Therefore we pre-divide the identity input by tp_size to cancel the extra scaling.
    """

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
        input_tensor = args[0]
        return (input_tensor / mesh.size(), *args[1:]), kwargs


class EpRouterParallel(TensorParallelLayer):
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
        if torch.is_grad_enabled() and router_scores.requires_grad:
            process_group = mesh.get_group() if mesh.ndim == 1 else mesh.get_group("tp")
            router_scores = _AllReduceBackward.apply(router_scores, process_group)
        non_local_mask = (router_indices // num_local_experts) != ep_rank
        router_scores = router_scores.masked_fill(non_local_mask, 0.0)
        router_indices = router_indices.masked_fill(non_local_mask, -1)
        if num_local_experts > 1:
            router_indices = torch.fmod(router_indices, num_local_experts)
        else:
            router_indices = router_indices.masked_fill(router_indices > 0, 0).masked_fill(router_indices < 0, -1)
        router_indices = router_indices.masked_fill(router_indices == -1, num_local_experts)
        return router_logits, router_scores, router_indices, *extra_outputs


class EpDispatchExpertsParallel(MoeExpertsParallel):
    """
    Expert parallelism with whole experts per rank, reached through an `EpStrategy` from `ALL_EP_STRATEGIES`.

    The style shards the experts once, whole experts per rank (`Shard(0)` on the EP mesh), and FSDP shards them
    further across the replicas of an expert (the `efsdp` mesh). How the tokens reach them is the strategy's job, named
    by `DistributedConfig.ep_implementation`: `"all_to_all"` sends each (token, expert) pair to the expert's rank,
    `"all_reduce"` masks every rank's experts on a batch the whole EP group holds and sums the outputs, and `"auto"`
    picks between the two on every forward. Every strategy runs on the same expert shards, so the choice can change
    from one forward to the next.
    """

    def install_forward(self, module, ep_mesh, *, tp_mesh=None, distributed_config=None):
        """Experts stay whole (Shard(0) on `ep_mesh`); `tp_mesh` is only the group of ranks holding the same batch."""
        strategy_name = _resolve_ep_strategy(distributed_config) if distributed_config is not None else "auto"
        strategy = ALL_EP_STRATEGIES[strategy_name](
            module, ep_mesh, tp_mesh=tp_mesh, distributed_config=distributed_config
        )

        def ep_forward(hidden_states, top_k_index, top_k_weights, **kwargs):
            if kwargs or not isinstance(top_k_index, torch.Tensor):
                raise NotImplementedError(
                    f"{type(module).__name__} routes its tokens itself, which `ep_dispatch_experts` cannot exchange: "
                    "run it with the router-masked plan (`ep_router` + `moe_tp_experts`) and `ep_size == tp_size`."
                )
            if isinstance(hidden_states, DTensor):
                hidden_states = hidden_states.to_local()
            if isinstance(top_k_weights, DTensor):
                top_k_weights = top_k_weights.to_local()
            return strategy(hidden_states, top_k_index, top_k_weights)

        module.forward = ep_forward
        return module


def apply_expert_parallelism(model: nn.Module, ep_mesh: DeviceMesh, tp_mesh: DeviceMesh, plan: dict[str, str]):
    """Shard experts on EP; use TP to split shared tokens and reconstruct outputs around dispatch."""
    for name, module in model.named_modules():
        for p_name, _ in list(module.named_parameters(recurse=False)):
            full = f"{name}.{p_name}" if name else p_name
            style_name = _get_parameter_plan(parameter_name=full, plan=plan, is_weight=True)
            if style_name is not None and style_name in ALL_PARALLEL_STYLES:
                style = ALL_PARALLEL_STYLES[style_name]
                style.validate_param(module, p_name, ep_mesh, parameter_name=full)
                style.shard_param(module, p_name, ep_mesh)

        # Dispatch hooks need both meshes to redistribute tokens between TP and EP ranks.
        style_name = _get_parameter_plan(parameter_name=name, plan=plan, is_weight=False)
        if style_name is not None and style_name in ALL_PARALLEL_STYLES:
            style = ALL_PARALLEL_STYLES[style_name]
            if isinstance(style, EpDispatchExpertsParallel):
                distributed_config = getattr(model.config, "distributed_config", None)
                style.install_forward(module, ep_mesh, tp_mesh=tp_mesh, distributed_config=distributed_config)
            else:
                style.install_forward(module, ep_mesh)
        module._is_hooked = True

    return model


if is_torch_distributed_available():
    ALL_PARALLEL_STYLES.register("ep_router", EpRouterParallel())
    ALL_PARALLEL_STYLES.register("ep_dispatch_experts", EpDispatchExpertsParallel())
    ALL_PARALLEL_STYLES.register("moe_tp_experts", MoeExpertsParallel())
    ALL_PARALLEL_STYLES.register("moe_identity_expert", MoeIdentityParallel())
