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
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..utils import logging
from ..utils.import_utils import is_torch_available, is_torch_distributed_available
from .parallel_utils import ALL_PARALLEL_STYLES, _use_local_dtensor_params
from .tensor_parallel import _get_parameter_plan


if TYPE_CHECKING:
    from torch import nn
    from torch.distributed.device_mesh import DeviceMesh


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

if is_torch_distributed_available():
    from torch.distributed._functional_collectives import all_to_all_single_autograd
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

    from .parallel_utils import _AllReduceBackward, _WaitTensor


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


class RouterParallelMegaMoe(EpRouterParallel):
    """Router TP plan used with DeepGEMM Mega MoE.

    Mega MoE handles EP dispatch inside the kernel and wants raw global expert ids
    with unmasked routing weights, so the router doesn't pre-shard per EP rank like
    ``EpRouterParallel`` does.
    """

    def transform_output_post_forward(self, module, output, mesh):
        return output


class EpDispatchExpertsParallel(ExpertParallelLayer):
    """
    Dispatch disjoint TP token slices to the experts' owners, then replicate the combined output on TP.

    Example:
    Let's say we have 8 experts [E0, E7] with DistributedConfig(tp_size=2, fsdp_size=4, ep_size=4). That imply:
        - Since fsdp_size=4, we have 4 batches B denoted [B0, B3]
        - Because we have tp_size=2, that means *both ranks share the same batch*
        - efsdp = (fsdp_size * tp_size) / ep_size = 4 * 2 / 4 = 2

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

    Assume token 1 in B0 chose E4, which lives on another rank, so it must travel by all-to-all. B0 sits on 2 ranks (cf diagram), so if both sent it, E4 would compute it twice.
    We need to make sure that token 1 is in rank 0 range, so rank 0 sends it and rank 1 does not have it in its slice.
    """

    def should_use_local_tensors(self, module):
        return True

    def _dispatch_tokens(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        num_local_experts: int,
        ep_group,
        ep_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
        """Send each selected (token, expert) pair to the rank that owns the expert.

        Also returns the sort order and the per-rank split sizes that `_combine_tokens` needs to reverse the exchange.
        """
        num_top_k = top_k_index.size(-1)

        # Sorting the selected pairs by expert groups them by owner rank, since each rank owns a contiguous range of
        # experts, and the per-expert counts tell every receiver which expert each token it gets is for. The split
        # sizes are the one host sync of the layer.
        expert_ids = top_k_index.reshape(-1)
        order = torch.argsort(expert_ids)
        send_tokens = hidden_states[order // num_top_k]
        send_counts = torch.zeros(num_local_experts * ep_size, dtype=torch.long, device=hidden_states.device)
        send_counts = send_counts.scatter_add_(0, expert_ids, torch.ones_like(expert_ids)).view(
            ep_size, num_local_experts
        )
        recv_counts = torch.empty_like(send_counts)
        torch.distributed.all_to_all_single(recv_counts, send_counts, group=ep_group)
        send_sizes, recv_sizes = torch.stack([send_counts.sum(dim=1), recv_counts.sum(dim=1)]).tolist()
        recv_tokens = all_to_all_single_autograd(send_tokens, recv_sizes, send_sizes, ep_group)
        recv_expert_ids = torch.arange(num_local_experts, device=hidden_states.device).repeat(ep_size)
        recv_expert_ids = recv_expert_ids.repeat_interleave(recv_counts.reshape(-1), output_size=sum(recv_sizes))
        return recv_tokens, recv_expert_ids, order, send_sizes, recv_sizes

    def _run_local_experts(
        self,
        experts_forward: Callable,
        tokens: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run local experts with top-1 routing and unit weights; apply routing weights after combine."""
        num_tokens = tokens.shape[0]
        if num_tokens == 0:
            # Keep the empty tokens connected to backward so the reverse all-to-all and FSDP reduction still run.
            dummy = tokens.sum(dim=0, keepdim=True)
            dummy_ids = torch.zeros((1, 1), dtype=expert_ids.dtype, device=tokens.device)
            dummy_weights = torch.ones((1, 1), dtype=tokens.dtype, device=tokens.device)
            return experts_forward(dummy, dummy_ids, dummy_weights)[:0]

        expert_ids = expert_ids.unsqueeze(-1)
        weights = torch.ones_like(expert_ids, dtype=tokens.dtype)
        return experts_forward(tokens, expert_ids, weights)

    def _combine_tokens(
        self,
        expert_output: torch.Tensor,
        top_k_weights: torch.Tensor,
        order: torch.Tensor,
        send_sizes: list[int],
        recv_sizes: list[int],
        ep_group,
    ) -> torch.Tensor:
        """Return expert outputs to the token owners and combine them with routing weights."""
        num_tokens, num_top_k = top_k_weights.shape
        hidden_dim = expert_output.size(-1)
        recv_out = all_to_all_single_autograd(expert_output, send_sizes, recv_sizes, ep_group)
        # Restore the original (token, top-k slot) order, then apply routing weights.
        token_outputs = torch.empty_like(recv_out)
        token_outputs[order] = recv_out
        token_outputs = token_outputs.view(num_tokens, num_top_k, hidden_dim)
        return (token_outputs * top_k_weights.unsqueeze(-1)).sum(dim=1)

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh, *, tp_mesh=None):
        hidden_states, top_k_index, top_k_weights = args
        if isinstance(hidden_states, DTensor):
            hidden_states = hidden_states.to_local()
        if isinstance(top_k_weights, DTensor):
            top_k_weights = top_k_weights.to_local()
        if tp_mesh is None or tp_mesh.size() == 1:
            return (hidden_states, top_k_index, top_k_weights), kwargs
        # TP ranks share the same batch, so keep only this rank's rows and each token is dispatched once.
        # Replicate -> Shard(0) is a local chunk (no communication); its backward all-gathers the row gradients.
        hidden_states = DTensor.from_local(hidden_states, tp_mesh, [Replicate()], run_check=False)
        top_k_index = DTensor.from_local(top_k_index, tp_mesh, [Replicate()], run_check=False)
        top_k_weights = DTensor.from_local(top_k_weights, tp_mesh, [Replicate()], run_check=False)

        hidden_states = hidden_states.redistribute(tp_mesh, [Shard(0)]).to_local()
        top_k_index = top_k_index.redistribute(tp_mesh, [Shard(0)]).to_local()
        top_k_weights = top_k_weights.redistribute(tp_mesh, [Shard(0)]).to_local()
        return (hidden_states, top_k_index, top_k_weights), kwargs

    def transform_output_post_forward(self, module, output, mesh, *, tp_mesh=None, num_tokens=None):
        if tp_mesh is None or tp_mesh.size() == 1:
            return output
        # Shard(0) -> Replicate is one all-gather of the row slices, which also handles uneven and empty slices.
        hidden_dim = output.size(-1)
        output = DTensor.from_local(
            output.contiguous(), tp_mesh, [Shard(0)], shape=(num_tokens, hidden_dim), stride=(hidden_dim, 1)
        )
        return output.full_tensor()

    def install_forward(self, module, ep_mesh, *, tp_mesh=None):
        """Experts stay whole (Shard(0) on `ep_mesh`); `tp_mesh` is only the group of ranks holding the same batch."""
        experts_forward = module.forward
        ep_group, ep_size = ep_mesh.get_group(), ep_mesh.size()

        def ep_forward(hidden_states, top_k_index, top_k_weights):
            # Read the full token count before the pre hook slices the inputs across the batch replicas.
            num_tokens = hidden_states.size(0)
            (hidden_states, top_k_index, top_k_weights), _ = self.transform_inputs_pre_forward(
                module, (hidden_states, top_k_index, top_k_weights), {}, ep_mesh, tp_mesh=tp_mesh
            )
            with self.context_around_forward(module, ep_mesh):
                tokens, expert_ids, order, send_sizes, recv_sizes = self._dispatch_tokens(
                    hidden_states, top_k_index, module.num_experts, ep_group, ep_size
                )
                # hand the experts the received rows, not the all-to-all's pending result (whose pointer is null until
                # waited on): https://github.com/huggingface/transformers/pull/49421
                tokens = _WaitTensor.apply(tokens)
                expert_output = self._run_local_experts(experts_forward, tokens, expert_ids)
                output = self._combine_tokens(
                    expert_output, top_k_weights, order, send_sizes, recv_sizes, ep_group
                ).to(hidden_states.dtype)

            return self.transform_output_post_forward(module, output, ep_mesh, tp_mesh=tp_mesh, num_tokens=num_tokens)

        module.forward = ep_forward
        return module


class MoeTensorParalellMegaMoeExperts(ExpertParallelLayer):
    """TP layer for DeepGEMM Mega MoE experts.

    Mega MoE is inference-only (the kernel has no backward) and handles EP dispatch +
    combine + per-rank token sharding internally — so we skip the gradient-sync hooks
    that ``MoeExpertsParallel`` would apply, and we forward the EP ``process_group``
    into the module so the symm-buffer rendezvous can run on first forward.
    """

    def transform_inputs_pre_forward(self, module, args, kwargs, mesh, *, is_expert_parallel=False):
        hidden_states, top_k_index, top_k_weights = args[0], args[1], args[2]
        return (hidden_states, top_k_index, top_k_weights, mesh.get_group()), kwargs

    def context_around_forward(self, module, mesh):
        return _use_local_dtensor_params(module)

    def transform_output_post_forward(self, module, output, mesh, source=None):
        return output


if is_torch_distributed_available():
    ALL_PARALLEL_STYLES.register("grouped_gemm", MoEParamShard(Shard(0), shards_expert_dim=True))
    ALL_PARALLEL_STYLES.register("ep_router", EpRouterParallel())
    ALL_PARALLEL_STYLES.register("ep_dispatch_experts", EpDispatchExpertsParallel())
    ALL_PARALLEL_STYLES.register("megamoe_router", RouterParallelMegaMoe())
    ALL_PARALLEL_STYLES.register("megamoe_experts", MoeTensorParalellMegaMoeExperts())


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
            if style_name == "ep_dispatch_experts":
                style.install_forward(module, ep_mesh=ep_mesh, tp_mesh=tp_mesh)
            else:
                style.install_forward(module, ep_mesh)
        module._is_hooked = True

    return model
