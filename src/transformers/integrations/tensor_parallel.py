# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.distributed.tensor.parallel.style import ParallelStyle

from ..utils import logging
from ..utils.import_utils import is_torch_available


if is_torch_available():
    import torch

    # Cache this result has it's a C FFI call which can be pretty time-consuming
    _torch_distributed_available = torch.distributed.is_available()


logger = logging.get_logger(__name__)


def replace_layer_number_by_wildcard(name: str) -> str:
    """
    Replace the numbers in the `name` by wildcards, only if they are in-between dots (`.`) or if they are between
    a dot (`.`) and the end of the string.
    This matches how modules are named/numbered when using a nn.ModuleList or nn.Sequential, but will NOT match
    numbers in a parameter name itself, e.g. if the param is named `"w1"` or `"w2"`.
    """
    return re.sub(r"\.\d+(\.|$)", lambda m: ".*" + m.group(1), name)


def _get_parameter_tp_plan(parameter_name: str, tp_plan: dict[str, str], is_weight=True) -> str | None:
    """
    Get the TP style for a parameter from the TP plan.

    The TP plan is a dictionary that maps parameter names to TP styles.
    The parameter name can be a generic name with wildcards (e.g. "*.weight") or a specific name (e.g. "layer_1.weight").

    The `is_weight` is important because for weights, we want to support `.weights` and `.bias` cases seamlessly! but
    not parent classes for `post_init` calls
    """
    generic_param_name = replace_layer_number_by_wildcard(parameter_name)
    if generic_param_name in tp_plan:
        return tp_plan[generic_param_name]
    elif is_weight and "." in generic_param_name and (module_name := generic_param_name.rsplit(".", 1)[0]) in tp_plan:
        return tp_plan[module_name]
    return None


# =============================================================================
# Tensor Sharding Utilities
# =============================================================================


def _to_cpu_fresh(tensor: torch.Tensor) -> torch.Tensor:
    """Plain tensor → contiguous CPU tensor with fresh storage for safetensors."""
    if tensor.device.type == "meta":
        return tensor
    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.to(device="cpu")
    out = torch.empty(t.shape, dtype=t.dtype, device="cpu")
    out.copy_(t)
    return out.contiguous()


# =============================================================================
# High-Level API Functions
# =============================================================================


def verify_tp_plan(expected_keys: list[str], tp_plan: dict[str, str | TPStyle] | None):
    """
    Verify the TP plan of the model, log a warning if the layers that were not sharded and the rules that were not applied.

    Only weight-sharding rules (colwise, rowwise, vocab) are checked.
    """

    if tp_plan is None:
        return

    generic_keys = {replace_layer_number_by_wildcard(key) for key in expected_keys}
    unsharded_layers = set(generic_keys)
    unused_rules = tp_plan.copy()

    for key in generic_keys:
        param_name = key.rsplit(".", 1)[0] if "." in key else key
        generic_param_name = re.sub(r"\d+", "*", param_name)

        if generic_param_name in tp_plan:
            unused_rules.pop(generic_param_name, None)
            unsharded_layers.discard(key)
        elif "." in generic_param_name and (parent_param_name := generic_param_name.rsplit(".", 1)[0]) in tp_plan:
            unused_rules.pop(parent_param_name, None)
            unsharded_layers.discard(key)

    if len(unused_rules) > 0:
        logger.warning(f"The following TP rules were not applied on any of the layers: {unused_rules}")
    if len(unsharded_layers) > 0:
        logger.warning(f"The following layers were not sharded: {', '.join(unsharded_layers)}")


@dataclass(frozen=True)
class TPStyle:
    kind: Literal["colwise", "rowwise", "vocab"]
    comm: Literal["none", "allreduce", "reduce_scatter"]
    sequence_dim: int = 1
    use_local_output: bool = True

    def to_dtensor_style(self) -> ParallelStyle:
        """Convert to the corresponding PyTorch DTensor ParallelStyle."""
        if self.kind == "colwise":
            match self.comm:
                case "none":
                    return ColwiseParallel(
                        input_layouts=Replicate(), output_layouts=Shard(-1), use_local_output=self.use_local_output
                    )
        elif self.kind == "rowwise":
            match self.comm:
                case "allreduce":
                    return RowwiseParallel(
                        input_layouts=Shard(-1),
                        output_layouts=Replicate(),
                        use_local_output=self.use_local_output,
                    )
                case "reduce_scatter":
                    return RowwiseParallel(
                        input_layouts=Shard(-1),
                        output_layouts=Shard(1),
                        use_local_output=self.use_local_output,
                    )
        elif self.kind == "vocab":
            match self.comm:
                case "allreduce":
                    return RowwiseParallel(
                        input_layouts=Replicate(),
                        output_layouts=Replicate(),
                        use_local_output=self.use_local_output,
                    )
                case "reduce_scatter":
                    return RowwiseParallel(
                        input_layouts=Replicate(), output_layouts=Shard(1), use_local_output=self.use_local_output
                    )
        raise ValueError(
            f"Invalid TPStyle({self.kind!r}, {self.comm!r}). Valid combinations:\n"
            f"  colwise:     none\n"
            f"  rowwise:     allreduce, reduce_scatter\n"
            f"  vocab:       allreduce, reduce_scatter"
        )

    def __str__(self):
        if self.comm == "none":
            return self.kind
        return f"{self.kind}_{self.comm}"


def apply_tensor_parallel(model, tp_mesh, tp_plan):
    """Apply tensor parallelism using PyTorch's parallelize_module.

    Converts the wildcard tp_plan from model config into a concrete plan
    for ``parallelize_module``. Plan values are ``TPStyle`` instances.
    """
    if tp_plan is None:
        return model

    if tp_plan == "auto":
        base_plan = model.config.base_model_tp_plan or {}

        # Prefix base model keys (e.g. "layers.*.q_proj" → "model.layers.*.q_proj")
        # Top-level keys like "lm_head" are kept as-is.
        base_model_prefix = model.base_model_prefix
        tp_plan = {}
        for k, v in base_plan.items():
            is_top_level = hasattr(model, k.split(".")[0])
            tp_plan[k if is_top_level else f"{base_model_prefix}.{k}"] = v

    parallelize_plan = {}

    for name, _ in model.named_modules():
        style_value = _get_parameter_tp_plan(parameter_name=name, tp_plan=tp_plan, is_weight=False)
        if style_value is None:
            continue

        if isinstance(style_value, TPStyle):
            dtensor_style = style_value.to_dtensor_style()
            parallelize_plan[name] = dtensor_style
        else:
            parallelize_plan[name] = style_value

    parallelize_module(model, tp_mesh, parallelize_plan)

    return model
