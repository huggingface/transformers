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
"""Backward-compatibility shim for the tensor parallel API.

The canonical implementation lives in ``transformers.distributed.tensor_parallel``.
"""

from __future__ import annotations

import warnings

from ..distributed.tensor_parallel import (
    ALL_PARALLEL_STYLES,
    AllReduceParallel,
    ColwiseParallel,
    EpRouterParallel,
    MlaKvAProjParallel,
    MoeExpertsParallel,
    MoeIdentityParallel,
    MoEParamShard,
    MoeTensorParalellMegaMoeExperts,
    PackedColwiseParallel,
    PackedRowwiseParallel,
    ParallelInterface,
    ReplicatedWithGradAllReduce,
    RouterParallelMegaMoe,
    RowwiseParallel,
    SequenceParallel,
    TensorParallelLayer,
    apply_tensor_parallelism,
    gather_state_dict_for_save,
    replace_layer_number_by_wildcard,
    verify_tp_plan,
)
from ..distributed.utils import initialize_tensor_parallelism


RouterParallel = EpRouterParallel
MoeIdentityExpertParallel = MoeIdentityParallel
MoeTensorParalellExperts = MoeExpertsParallel


def shard_and_distribute_module(*args, **kwargs):
    """Deprecated per-parameter sharding helper from the legacy TP loading path."""
    warnings.warn(
        "`shard_and_distribute_module` is deprecated and unavailable with the DTensor tensor-parallel "
        "loading path. Use `transformers.distributed.tensor_parallel.apply_tensor_parallelism` with "
        "`from_pretrained(..., tp_plan=...)` instead.",
        FutureWarning,
        stacklevel=2,
    )
    raise RuntimeError("`shard_and_distribute_module` is unavailable with the DTensor tensor-parallel loading path.")


__all__ = [
    "ALL_PARALLEL_STYLES",
    "MoeIdentityExpertParallel",
    "MoeTensorParalellExperts",
    "RouterParallel",
    "initialize_tensor_parallelism",
    "AllReduceParallel",
    "ColwiseParallel",
    "EpRouterParallel",
    "MlaKvAProjParallel",
    "MoeExpertsParallel",
    "MoEParamShard",
    "MoeIdentityParallel",
    "MoeTensorParalellMegaMoeExperts",
    "PackedColwiseParallel",
    "PackedRowwiseParallel",
    "ParallelInterface",
    "ReplicatedWithGradAllReduce",
    "RouterParallelMegaMoe",
    "RowwiseParallel",
    "SequenceParallel",
    "TensorParallelLayer",
    "apply_tensor_parallelism",
    "gather_state_dict_for_save",
    "replace_layer_number_by_wildcard",
    "shard_and_distribute_module",
    "verify_tp_plan",
]
