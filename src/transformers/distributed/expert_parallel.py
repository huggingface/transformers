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
from .parallel_utils import ALL_PARALLEL_STYLES, _use_local_dtensor_params


if TYPE_CHECKING:
    from torch import nn
    from torch.distributed.device_mesh import DeviceMesh


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

if is_torch_distributed_available():
    from torch.distributed import all_gather_into_tensor, all_to_all_single, get_process_group_ranks
    from torch.distributed._functional_collectives import all_to_all_single_autograd
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

    from .parallel_utils import _AllReduceBackward, _AllReduceForward, _WaitTensor


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

    Shards dim 0 and updates module.num_experts to the per-rank local count, so the experts forward skips the
    expert ids the EP implementations mark as off-rank.
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


class EPImplementation:
    """
    How the experts of an `ep_dispatch_experts` module get their tokens. One instance is built per experts module when
    the plan is applied, so an implementation can keep state across forwards (communication buffers, a layout it
    reuses).

    Calling it with `(hidden_states, top_k_index, top_k_weights)`, the batch this rank holds (global expert ids),
    returns the weighted expert output for each of its tokens. It runs this rank's experts with `experts_forward`, on
    whatever rows, local expert ids and weights it brought them.

    Gradients must reach every rank that holds a token, routing weights included, so a collective that autograd does not
    differentiate has to be wrapped in a `torch.autograd.Function`.
    """

    # Set by implementations that only run when every rank of the EP group holds the same batch, e.g. `AllReduceEP`.
    needs_shared_batch = False

    def __init__(self, experts: nn.Module, ep_mesh: DeviceMesh, tp_mesh: DeviceMesh):
        self.experts_forward = experts.forward  # the experts' own forward, on local expert ids
        self.num_local_experts = experts.num_experts
        self.ep_mesh = ep_mesh
        self.tp_mesh = tp_mesh
        # Every rank of the EP group holds the same batch when the EP group is the TP group.
        tp_ranks, ep_ranks = (get_process_group_ranks(mesh.get_group()) for mesh in (tp_mesh, ep_mesh))
        self.shares_batch = tp_ranks == ep_ranks
        if self.needs_shared_batch and not self.shares_batch:
            raise ValueError(
                f"{type(self).__name__} needs every rank of the expert group to hold the same batch (`ep_size == tp_size`)."
            )

    def __call__(self, hidden_states, top_k_index, top_k_weights):
        raise NotImplementedError


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

    def __call__(self, hidden_states, top_k_index, top_k_weights):
        tokens, expert_ids, weights, state = self._dispatch_tokens(hidden_states, top_k_index, top_k_weights)
        if tokens.shape[0] == 0:
            # Keep the empty rows connected to backward, so the reverse exchange and FSDP's reduction still run.
            expert_output = self.experts_forward(
                tokens.sum(dim=0, keepdim=True), expert_ids.new_zeros((1, 1)), weights.new_ones((1, 1))
            )[:0]
        else:
            expert_output = self.experts_forward(tokens, expert_ids, weights)
        return self._combine_tokens(expert_output, state)

    def _dispatch_tokens(self, hidden_states, top_k_index, top_k_weights):
        """Each (token, expert) pair to the expert's rank: the rows this rank's experts run on, their local expert ids
        and weights, and what `_combine_tokens` needs to bring the results back."""
        num_tokens = hidden_states.size(0)
        if self.tp_mesh.size() > 1:
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
        all_to_all_single(recv_counts, send_counts, group=ep_group)
        send_sizes, recv_sizes = torch.stack([send_counts.sum(dim=1), recv_counts.sum(dim=1)]).tolist()
        recv_tokens = all_to_all_single_autograd(send_tokens, recv_sizes, send_sizes, ep_group)
        # hand the experts the received rows, not the all-to-all's pending result (whose pointer is null until
        # waited on): https://github.com/huggingface/transformers/pull/49421
        recv_tokens = _WaitTensor.apply(recv_tokens)
        recv_expert_ids = torch.arange(self.num_local_experts, device=hidden_states.device).repeat(ep_size)
        recv_expert_ids = recv_expert_ids.repeat_interleave(recv_counts.reshape(-1), output_size=sum(recv_sizes))
        recv_expert_ids = recv_expert_ids.unsqueeze(-1)
        # Experts run top-1 with unit weights; the routing weights are applied in `_combine_tokens`, on the sender.
        recv_weights = torch.ones_like(recv_expert_ids, dtype=recv_tokens.dtype)
        state = _AllToAllState(top_k_weights, order, send_sizes, recv_sizes, num_tokens)
        return recv_tokens, recv_expert_ids, recv_weights, state

    def _combine_tokens(self, expert_output, state):
        """The expert outputs back to the ranks that sent the tokens, weighted and summed per token."""
        hidden_dim = expert_output.size(-1)
        recv_out = all_to_all_single_autograd(
            expert_output, state.send_sizes, state.recv_sizes, self.ep_mesh.get_group()
        )
        # Restore the original (token, top-k slot) order, then apply routing weights.
        token_outputs = torch.empty_like(recv_out)
        token_outputs[state.order] = recv_out
        token_outputs = token_outputs.view(*state.top_k_weights.shape, hidden_dim)
        output = (token_outputs * state.top_k_weights.unsqueeze(-1)).sum(dim=1)
        if self.tp_mesh.size() == 1:
            return output
        # Shard(0) -> Replicate is one all-gather of the row slices, uneven or empty ones too.
        output = DTensor.from_local(
            output.contiguous(),
            self.tp_mesh,
            [Shard(0)],
            shape=(state.num_tokens, hidden_dim),
            stride=(hidden_dim, 1),
        )
        return output.full_tensor()


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

    def __call__(self, hidden_states, top_k_index, top_k_weights):
        ep_group = self.ep_mesh.get_group()
        # Each rank backpropagates through its own experts only, so the gradients of the shared inputs it computes are
        # partial: sum them over the group (the forward all-reduce has an identity backward).
        hidden_states = _AllReduceBackward.apply(hidden_states, ep_group)
        top_k_weights = _AllReduceBackward.apply(top_k_weights, ep_group)
        off_rank = (top_k_index // self.num_local_experts) != self.ep_mesh.get_local_rank()
        local_index = torch.where(off_rank, self.num_local_experts, top_k_index % self.num_local_experts)
        expert_output = self.experts_forward(hidden_states, local_index, top_k_weights.masked_fill(off_rank, 0.0))
        return _AllReduceForward.apply(expert_output, ep_group)


def _uses_megamoe(experts: nn.Module) -> bool:
    """Mega MoE experts exchange the tokens inside their kernel, so only `MegaMoeEP` can run them."""
    return getattr(getattr(experts, "config", None), "_experts_implementation", None) == "deepgemm_megamoe"


class AutoEP(EPImplementation):
    """
    The default. It runs, in order:

    - `MegaMoeEP` for DeepGEMM Mega MoE experts (`experts_implementation="deepgemm_megamoe"`), which only run
      through it.
    - `AllReduceEP` at inference (no gradients through the inputs) when the EP group shares one batch. It never reads
      split sizes back to the host, so decode captures in a CUDA graph.
    - `AllToAllEP` otherwise: in training, and wherever the ranks hold different batches.
    """

    def __init__(self, experts: nn.Module, ep_mesh: DeviceMesh, tp_mesh: DeviceMesh):
        super().__init__(experts, ep_mesh, tp_mesh)
        self.all_to_all = AllToAllEP(experts, ep_mesh, tp_mesh)
        self.all_reduce = AllReduceEP(experts, ep_mesh, tp_mesh) if self.shares_batch else None
        self.megamoe = MegaMoeEP(experts, ep_mesh, tp_mesh) if _uses_megamoe(experts) else None

    def __call__(self, hidden_states, top_k_index, top_k_weights):
        if self.megamoe is not None:
            return self.megamoe(hidden_states, top_k_index, top_k_weights)
        if self.all_reduce is not None and not (torch.is_grad_enabled() and hidden_states.requires_grad):
            return self.all_reduce(hidden_states, top_k_index, top_k_weights)
        return self.all_to_all(hidden_states, top_k_index, top_k_weights)


class MegaMoeEP(EPImplementation):
    """
    DeepGEMM Mega MoE, whose kernel fuses the token dispatch, the experts and the combine. It treats each rank's rows as
    that rank's own tokens. Here every rank of the EP group holds the same batch, so each rank hands the kernel an equal
    slice of it, with global expert ids and unmasked weights, and the slices' outputs are gathered back. Every token goes
    through the experts once. The kernel has no backward, so this runs at inference only.

    Example: 4 tokens on 2 ranks.

    GPU                    0                      1
                           |                      |
    batch           [t0, t1, t2, t3]       [t0, t1, t2, t3]     the same tokens on both ranks
    kernel input        [t0, t1]               [t2, t3]          each rank's slice
    kernel output    [y0, y1]               [y2, y3]            the kernel exchanges tokens between ranks
    after all-gather         [y0, y1, y2, y3] on both ranks
    """

    # TODO: Mega MoE is built for ranks holding different batches (data-parallel attention + EP), where neither the
    # slicing nor the gather is needed. Until inference supports data parallelism, it runs on the shared batch.
    needs_shared_batch = True

    def __call__(self, hidden_states, top_k_index, top_k_weights):
        ep_group, ep_size, ep_rank = self.ep_mesh.get_group(), self.ep_mesh.size(), self.ep_mesh.get_local_rank()
        num_tokens = hidden_states.size(0)
        # equal slices so the outputs gather evenly; a slice running past the batch is padded with skipped slots
        # (expert id -1, weight 0), only on the ranks it reaches
        rows_per_rank = max(1, -(-num_tokens // ep_size))
        rows = slice(ep_rank * rows_per_rank, (ep_rank + 1) * rows_per_rank)
        hidden_states, top_k_index, top_k_weights = hidden_states[rows], top_k_index[rows], top_k_weights[rows]
        padding = rows_per_rank - hidden_states.size(0)
        if padding:
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, padding))
            top_k_index = torch.nn.functional.pad(top_k_index, (0, 0, 0, padding), value=-1)
            top_k_weights = torch.nn.functional.pad(top_k_weights, (0, 0, 0, padding))

        local_output = self.experts_forward(hidden_states, top_k_index, top_k_weights, process_group=ep_group)
        output = local_output.new_empty(rows_per_rank * ep_size, local_output.size(-1))
        all_gather_into_tensor(output, local_output.contiguous(), group=ep_group)
        return output[:num_tokens].to(hidden_states.dtype)


class EPImplementationInterface(GeneralInterface):
    """Registry of the `EPImplementation`s `DistributedConfig.ep_implementation` names. Add one with
    `EPImplementationInterface.register(name, cls)`."""

    _global_mapping = {
        "auto": AutoEP,
        "all_to_all": AllToAllEP,
        "all_reduce": AllReduceEP,
        "deepgemm_megamoe": MegaMoeEP,
    }


class EpDispatchExpertsParallel(ExpertParallelLayer):
    """Whole experts per rank (`Shard(0)` on the EP mesh), reached through the `EPImplementation` that
    `DistributedConfig.ep_implementation` names in `EPImplementationInterface`."""

    def should_use_local_tensors(self, module):
        return True

    def install_forward(self, module, ep_mesh, *, tp_mesh, ep_implementation="auto"):
        """`tp_mesh` is the group of ranks holding the same batch (size 1 without TP)."""
        if _uses_megamoe(module) and ep_implementation not in ("auto", "deepgemm_megamoe"):
            raise ValueError(
                f'`experts_implementation="deepgemm_megamoe"` runs with `ep_implementation="auto"` or '
                f'`"deepgemm_megamoe"`, got `ep_implementation={ep_implementation!r}`.'
            )
        implementations = EPImplementationInterface()
        if ep_implementation not in implementations:
            raise ValueError(
                f"`ep_implementation={ep_implementation!r}` names no registered implementation: use one of "
                f"{', '.join(repr(name) for name in implementations)}, or register an `EPImplementation` "
                "with `EPImplementationInterface.register`."
            )
        implementation = implementations[ep_implementation](module, ep_mesh, tp_mesh)

        def ep_forward(hidden_states, top_k_index, top_k_weights):
            if isinstance(hidden_states, DTensor):
                hidden_states = hidden_states.to_local()
            if isinstance(top_k_weights, DTensor):
                top_k_weights = top_k_weights.to_local()
            with self.context_around_forward(module, ep_mesh):
                output = implementation(hidden_states, top_k_index, top_k_weights)
            return output.to(hidden_states.dtype)

        module.forward = ep_forward
        return module


if is_torch_distributed_available():
    ALL_PARALLEL_STYLES.register("grouped_gemm", MoEParamShard(Shard(0), shards_expert_dim=True))
    ALL_PARALLEL_STYLES.register("ep_dispatch_experts", EpDispatchExpertsParallel())


def apply_expert_parallelism(
    model: nn.Module,
    ep_mesh: DeviceMesh,
    tp_mesh: DeviceMesh,
    plan: dict[str, str],
    ep_implementation: str = "auto",
):
    """
    Shard experts on EP; `ep_dispatch_experts` modules reach them through the `EPImplementation` that
    `ep_implementation` names in `EPImplementationInterface`.
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
                style.install_forward(module, ep_mesh=ep_mesh, tp_mesh=tp_mesh, ep_implementation=ep_implementation)
            else:
                style.install_forward(module, ep_mesh)
        module._is_hooked = True

    return model
