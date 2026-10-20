# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Fine-grained quantizer for models whose modules use different block-scaled formats.

Each module loads through the quantizer of its own format.
"""

import re
from typing import TYPE_CHECKING

from .block_scaled_utils import BlockScaledQuantizer
from .quantizers_utils import (
    checkpoint_module_namer,
    get_module_from_name,
    module_kind,
    plan_entry_for,
)


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import FineGrainedConfig


class FineGrainedHfQuantizer(BlockScaledQuantizer):
    """Quantizer that hosts one member quantizer per format in the config's `dtype_plan`.

    Shared steps run once over the whole plan. Each module's tensors take the ops of its entry's quantizer.
    DeepSeek-V4 is an example, with MXFP4 experts and block FP8 linears.
    """

    quantization_config: "FineGrainedConfig"

    def __init__(self, quantization_config: "FineGrainedConfig", **kwargs):
        super().__init__(quantization_config, **kwargs)
        from .auto import AUTO_QUANTIZER_MAPPING

        self._resolved_plan = None
        # Equal entries share one config, so a per-layer export builds one member per format
        shared = {}
        quantization_config.dtype_plan = {
            pattern: shared.setdefault(config.to_json_string(), config)
            for pattern, config in quantization_config.dtype_plan.items()
        }
        configs = {id(config): config for config in self.plan().values()}.values()
        self._members = [AUTO_QUANTIZER_MAPPING[config.method.value](config, **kwargs) for config in configs]
        self._member_by_config = {id(member.quantization_config): member for member in self._members}
        for member in self._members:
            member.embeddings = self.embeddings

    def plan(self):
        dtype_plan = self.quantization_config.dtype_plan if self._resolved_plan is None else self._resolved_plan
        return {**self._embedding_plan(), **dtype_plan}

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        # Plan patterns may use checkpoint module names. The saved config keeps them.
        to_checkpoint = checkpoint_module_namer(model)
        plan = self.quantization_config.dtype_plan
        resolved = dict(plan)
        matched = set()
        for name, module in model.named_modules():
            if module_kind(name, module) is None:
                continue
            checkpoint_name = to_checkpoint(name)
            native = plan_entry_for(plan, name, catch_all=False)
            source = plan_entry_for(plan, checkpoint_name, catch_all=False)
            if native is not None and source is not None and native is not source:
                raise ValueError(
                    f"Conflicting quantization entries for {name!r} and checkpoint module {checkpoint_name!r}"
                )
            if source is not None and native is None:
                resolved[rf"^{re.escape(name)}$"] = source
            matched.update(
                pattern for pattern in plan if re.search(pattern, name) or re.search(pattern, checkpoint_name)
            )
        for pattern, config in plan.items():
            if config.checkpoint_format == "modelopt" and pattern not in matched:
                raise ValueError(
                    f"ModelOpt target {pattern!r} matches no supported linear, expert, or embedding module. "
                    "Per-expert assignments cannot target an architecture that stores fused expert tensors."
                )
        self._resolved_plan = resolved
        return super()._process_model_before_weight_loading(model, **kwargs)

    def members(self):
        return self._members

    def member_of(self, module) -> "BlockScaledQuantizer | None":
        config = getattr(module, "quantization_config", None)
        if config is None:
            return None
        member = self._member_by_config.get(id(config))
        if member is not None:
            return member
        # A module may hold an equal copy of its entry's config
        return next((m for m in self._members if m.quantization_config.to_dict() == config.to_dict()), None)

    def get_param_ops(self, model: "PreTrainedModel", param_name: str) -> list:
        member = self.member_of(get_module_from_name(model, param_name)[0])
        return member.get_param_ops(model, param_name) if member is not None else []
