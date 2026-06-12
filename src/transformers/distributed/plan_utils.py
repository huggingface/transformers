# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""Config-time helpers for building explicit combo parallel plans."""


def merge_dense_and_ep_plan(dense_plan: dict[str, str], ep_plan: dict[str, str]) -> dict[str, str]:
    """
    Build a complete TP+EP or SP+EP plan at config init time from dense and EP recipes.

    EP entries win on conflict; intra-expert ``moe_tp_*`` keys from the dense plan are dropped.
    """
    merged = {**dense_plan, **ep_plan}
    return {
        key: style
        for key, style in merged.items()
        if not (
            key.endswith((".gate_up_proj", ".down_proj"))
            and style in ("moe_tp_gate_up_colwise", "moe_tp_down_rowwise")
        )
    }
