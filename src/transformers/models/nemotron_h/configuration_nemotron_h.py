# Copyright 2024-2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
"""NemotronH backward-compat config dispatcher.

The original Nemotron-H config accepted any mix of mamba/attention/mlp/moe blocks,
but the released checkpoints only realise two architectures:

* ``nemotron_h_dense``  — mamba + attention + mlp (Nemotron-H, Nemotron-v2)
* ``nemotron_h_sparse`` — mamba + attention + moe (Nemotron-3)

This module keeps ``NemotronHConfig`` alive as a dispatcher so that existing
checkpoints on the Hub (``model_type: nemotron_h``) continue to load. Instantiating
``NemotronHConfig(...)`` or loading such a config returns either a
``NemotronHDenseConfig`` or a ``NemotronHSparseConfig`` based on the block pattern.
"""

from __future__ import annotations

from ...configuration_utils import PreTrainedConfig
from ...utils import logging
from ..nemotron_h_dense.configuration_nemotron_h_dense import NemotronHDenseConfig
from ..nemotron_h_sparse.configuration_nemotron_h_sparse import NemotronHSparseConfig


logger = logging.get_logger(__name__)


def _is_sparse_kwargs(kwargs: dict) -> bool:
    """Detect whether a raw config kwargs/dict is a sparse (MoE) Nemotron-H."""
    if "mtp_hybrid_override_pattern" in kwargs or kwargs.get("num_nextn_predict_layers"):
        return True
    pattern = kwargs.get("hybrid_override_pattern")
    if isinstance(pattern, str) and "E" in pattern:
        return True
    layers = kwargs.get("layers_block_type") or kwargs.get("layer_types")
    if layers and any(t == "moe" for t in layers):
        return True
    return False


def _resolve_nemotron_h_class(kwargs: dict) -> type[PreTrainedConfig]:
    return NemotronHSparseConfig if _is_sparse_kwargs(kwargs) else NemotronHDenseConfig


class NemotronHConfig(PreTrainedConfig):
    """Dispatcher shim for backward compatibility with ``model_type: nemotron_h``.

    Use :class:`NemotronHDenseConfig` or :class:`NemotronHSparseConfig` directly
    for new code. Instantiating this class redirects to the appropriate one
    based on whether the block pattern contains MoE (``E``) or MLP (``-``) blocks.
    """

    model_type = "nemotron_h"

    def __new__(cls, *args, **kwargs):
        if cls is NemotronHConfig:
            target_cls = _resolve_nemotron_h_class(kwargs)
            logger.info(
                f"`NemotronHConfig` has been split into `NemotronHDenseConfig` and `NemotronHSparseConfig`; "
                f"dispatching to `{target_cls.__name__}`."
            )
            return target_cls(*args, **kwargs)
        return super().__new__(cls)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        if cls is NemotronHConfig:
            target_cls = _resolve_nemotron_h_class(config_dict)
            return target_cls.from_dict(config_dict, **kwargs)
        return super().from_dict(config_dict, **kwargs)


__all__ = ["NemotronHConfig"]
