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
"""Resolution of the FSDP plan around parameters the loader keeps in another dtype.

These run on CPU against a meta-device model: they cover the plan that `apply_fully_sharded_data_parallelism`
consumes, which the distributed FSDP tests can only exercise on several accelerators.
"""

import unittest

from transformers import Ernie4_5_MoeConfig, LlamaConfig
from transformers.testing_utils import require_torch


try:
    import torch

    from transformers.distributed.configuration_utils import DistributedConfig
    from transformers.distributed.fsdp import _get_fsdp_policy_kwargs, _get_group_policy_kwargs
    from transformers.distributed.mixin import _apply_dtype_plan
except ImportError:
    pass


def _ernie_moe_model(dtype):
    """A tiny ERNIE-4.5-MoE: its router `gate.weight` and `moe_statics` are kept in float32 by the loader."""
    from transformers import Ernie4_5_MoeForCausalLM

    config = Ernie4_5_MoeConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        moe_num_experts=4,
        moe_k=2,
    )
    with torch.device("meta"):
        return Ernie4_5_MoeForCausalLM(config).to(dtype)


def _llama_model(dtype):
    """A tiny Llama: every parameter stays in the requested dtype."""
    from transformers import LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    with torch.device("meta"):
        return LlamaForCausalLM(config).to(dtype)


def _resolve(model, **config_kwargs):
    _apply_dtype_plan(model, model._get_dtype_plan(model.dtype))
    _, _, fsdp_plan = model._resolve_parallel_plans(DistributedConfig(fsdp_size=2, **config_kwargs))
    return fsdp_plan


def _group_dtypes(model, group_name, plan):
    """The dtypes FSDP2 would flatten into `group_name`'s buffer: its parameters minus those a nested group owns."""
    nested = [name for name in plan if name.startswith(f"{group_name}.")]
    return {
        param.dtype
        for name, param in model.get_submodule(group_name).named_parameters()
        if not any(f"{group_name}.{name}".startswith(f"{other}.") for other in nested)
    }


@require_torch
class FSDPDtypePlanTest(unittest.TestCase):
    def test_dtype_plan_is_applied_before_the_plan_is_built(self):
        """The loader upcasts after sharding, so the plan has to be built against the post-load dtypes."""
        model = _ernie_moe_model(torch.bfloat16)
        self.assertEqual({param.dtype for param in model.parameters()}, {torch.bfloat16})

        _apply_dtype_plan(model, model._get_dtype_plan(torch.bfloat16))

        upcast = {name for name, param in model.named_parameters() if param.dtype == torch.float32}
        self.assertTrue(upcast, "no parameter was upcast")
        for name in upcast:
            self.assertTrue(
                name.endswith("gate.weight") or "moe_statics" in name,
                f"{name} was upcast but is not in the model's dtype plan",
            )
        # Retyping happens on meta and must not move or unfreeze anything.
        self.assertTrue(all(param.device.type == "meta" for param in model.parameters()))

    def test_dtype_plan_does_not_change_the_model_dtype(self):
        """`model.dtype` reads the first floating point parameter; upcasting a router must not flip it."""
        model = _ernie_moe_model(torch.bfloat16)
        _apply_dtype_plan(model, model._get_dtype_plan(torch.bfloat16))
        self.assertEqual(model.dtype, torch.bfloat16)

    def test_every_group_holds_exactly_one_dtype(self):
        """FSDP2 all-gathers a group into one flat buffer and asserts on a group holding two dtypes."""
        model = _ernie_moe_model(torch.bfloat16)
        plan = _resolve(model)

        for group_name in plan:
            self.assertEqual(
                len(_group_dtypes(model, group_name, plan)),
                1,
                f"{group_name} would be all-gathered with more than one dtype",
            )

    def test_carved_groups_come_before_their_parent(self):
        """FSDP2 requires a nested group to be sharded before the group that contains it."""
        model = _ernie_moe_model(torch.bfloat16)
        plan = list(_resolve(model))

        carved = [name for name in plan if name.endswith("mlp.gate")]
        self.assertTrue(carved, "the float32 router was not carved out")
        for name in carved:
            parent = name.rsplit(".mlp.gate", 1)[0]
            self.assertIn(parent, plan)
            self.assertLess(plan.index(name), plan.index(parent), f"{name} is sharded after {parent}")

    def test_a_router_and_its_statics_share_one_group(self):
        """Both are kept in float32, so the statics need no group of their own inside the router's."""
        model = _ernie_moe_model(torch.bfloat16)
        plan = _resolve(model)
        self.assertFalse([name for name in plan if name.endswith("moe_statics")])

    def test_a_uniform_model_keeps_its_plan_unchanged(self):
        """Nothing is carved when no group holds more than one dtype."""
        model = _llama_model(torch.bfloat16)
        before = model._resolve_parallel_plans(DistributedConfig(fsdp_size=2))[2]
        after = _resolve(model)
        self.assertEqual(before, after)

    def test_a_frozen_parameter_of_another_dtype_does_not_split_a_group(self):
        """FSDP2 checks dtype uniformity over trainable parameters only, so a frozen outlier needs no group."""
        model = _llama_model(torch.bfloat16)
        frozen = model.model.layers[0].mlp.down_proj.weight
        frozen.data = frozen.data.to(torch.float32)
        frozen.requires_grad_(False)

        plan = _resolve(model)
        self.assertFalse([name for name in plan if name.endswith("down_proj")])

    def test_a_float32_model_is_not_split(self):
        """`_keep_in_fp32_modules_strict` is inert in float32: the model stays uniform and the plan untouched."""
        model = _ernie_moe_model(torch.float32)
        plan = _resolve(model)
        self.assertFalse([name for name in plan if name.endswith("mlp.gate")])


@require_torch
class FSDPGroupPolicyTest(unittest.TestCase):
    def _policy(self, model, group_name, **config_kwargs):
        base = _get_fsdp_policy_kwargs(DistributedConfig(fsdp_size=2, **config_kwargs))
        return _get_group_policy_kwargs(model.get_submodule(group_name), base)

    def test_a_carved_group_is_gathered_in_its_own_dtype(self):
        """Gathering it in `param_dtype` would silently undo the upcast `_keep_in_fp32_modules_strict` asked for."""
        model = _ernie_moe_model(torch.bfloat16)
        plan = _resolve(model)
        router = next(name for name in plan if name.endswith("mlp.gate"))

        kwargs = self._policy(model, router, fsdp_mixed_precision=True)
        self.assertEqual(kwargs["mp_policy"].param_dtype, torch.float32)

    def test_an_ordinary_group_keeps_the_configured_policy(self):
        model = _ernie_moe_model(torch.bfloat16)
        _resolve(model)

        kwargs = self._policy(model, "model.layers.0", fsdp_mixed_precision=True)
        self.assertEqual(kwargs["mp_policy"].param_dtype, torch.bfloat16)

    def test_mixed_precision_off_adds_no_policy(self):
        model = _ernie_moe_model(torch.bfloat16)
        plan = _resolve(model)
        router = next(name for name in plan if name.endswith("mlp.gate"))

        kwargs = self._policy(model, router)
        self.assertNotIn("mp_policy", kwargs)
