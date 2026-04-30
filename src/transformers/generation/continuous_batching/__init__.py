# Copyright 2025 The HuggingFace Inc. team
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
import os


# Disable NCCL's intra-process graph mixing protection: under CUDA-graph capture it spawns one stream per
# captured collective (~hundreds for TP forward) and inflates per-kernel launch overhead. Safe because CB
# enqueues collectives serially from a single generation thread. Must be set before init_process_group, so
# this lives at package import. ContinuousBatchingManager._maybe_warn_nccl_graph_mixing checks at runtime
# whether the value actually took effect and warns if NCCL was already initialized when this ran.
os.environ.setdefault("NCCL_GRAPH_MIXING_SUPPORT", "0")

from .cache import PagedAttentionCache
from .continuous_api import ContinuousBatchingManager, ContinuousMixin
from .requests import RequestState, RequestStatus
from .scheduler import FIFOScheduler, PrefillFirstScheduler, Scheduler


__all__ = [
    "ContinuousBatchingManager",
    "ContinuousMixin",
    "FIFOScheduler",
    "PagedAttentionCache",
    "PrefillFirstScheduler",
    "RequestState",
    "RequestStatus",
    "Scheduler",
]
