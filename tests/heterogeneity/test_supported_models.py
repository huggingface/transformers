# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import unittest
from collections.abc import Callable
from dataclasses import dataclass

from parameterized import parameterized

from transformers.testing_utils import cleanup, is_torch_available, require_torch, torch_device


HETERO_CASES = []

if is_torch_available():
    import torch

    from tests.heterogeneity.model_fixtures import MODEL_FIXTURES
    from tests.heterogeneity.testing_utils import (
        build_model,
        dummy_input_ids,
        forward_logits,
        tiny_gpt_oss_config,
        tiny_llama4_config,
        tiny_llama_config,
        tiny_nemotron_h_config,
    )
    from transformers import LlamaForCausalLM
    from transformers.integrations.heterogeneity import get_heterogeneous_modeling_specs
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
    from transformers.models.llama4.modeling_llama4 import Llama4ForCausalLM
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

    @dataclass(frozen=True)
    class HeteroCase:
        """Per-layer overrides for a supported model."""

        name: str
        model_key: str  # key into MODEL_FIXTURES
        config_factory: Callable
        model_cls: type
        per_layer_config: dict

    HETERO_CASES = [
        # ── Llama ──
        HeteroCase(
            name="llama_multi_attr",
            model_key="llama",
            config_factory=tiny_llama_config,
            model_cls=LlamaForCausalLM,
            per_layer_config={
                0: {"intermediate_size": 64, "num_key_value_heads": 2},
                2: {"intermediate_size": 96, "num_key_value_heads": 1},
            },
        ),
        HeteroCase(
            name="llama_skip_attn",
            model_key="llama",
            config_factory=tiny_llama_config,
            model_cls=LlamaForCausalLM,
            per_layer_config={1: {"skip": ["attention"]}},
        ),
        HeteroCase(
            name="llama_skip_mlp",
            model_key="llama",
            config_factory=tiny_llama_config,
            model_cls=LlamaForCausalLM,
            per_layer_config={2: {"skip": ["mlp"]}},
        ),
        HeteroCase(
            name="llama_skip_both",
            model_key="llama",
            config_factory=tiny_llama_config,
            model_cls=LlamaForCausalLM,
            per_layer_config={1: {"skip": ["attention", "mlp"]}},
        ),
        # ── GPT-OSS ──
        HeteroCase(
            name="gpt_oss_dim",
            model_key="gpt_oss",
            config_factory=tiny_gpt_oss_config,
            model_cls=GptOssForCausalLM,
            per_layer_config={0: {"intermediate_size": 16}, 2: {"intermediate_size": 48}},
        ),
        HeteroCase(
            name="gpt_oss_skip_attn",
            model_key="gpt_oss",
            config_factory=tiny_gpt_oss_config,
            model_cls=GptOssForCausalLM,
            per_layer_config={1: {"skip": ["attention"]}},
        ),
        HeteroCase(
            name="gpt_oss_skip_mlp",
            model_key="gpt_oss",
            config_factory=tiny_gpt_oss_config,
            model_cls=GptOssForCausalLM,
            per_layer_config={2: {"skip": ["mlp"]}},
        ),
        # ── Llama4 ──
        HeteroCase(
            name="llama4_dim",
            model_key="llama4",
            config_factory=tiny_llama4_config,
            model_cls=Llama4ForCausalLM,
            per_layer_config={0: {"intermediate_size_mlp": 64}},
        ),
        HeteroCase(
            name="llama4_skip_attn",
            model_key="llama4",
            config_factory=tiny_llama4_config,
            model_cls=Llama4ForCausalLM,
            per_layer_config={2: {"skip": ["attention"]}},
        ),
        HeteroCase(
            name="llama4_skip_mlp",
            model_key="llama4",
            config_factory=tiny_llama4_config,
            model_cls=Llama4ForCausalLM,
            per_layer_config={0: {"skip": ["mlp"]}},
        ),
        HeteroCase(
            name="llama4_skip_moe_mlp",
            model_key="llama4",
            config_factory=tiny_llama4_config,
            model_cls=Llama4ForCausalLM,
            per_layer_config={1: {"skip": ["mlp"]}},
        ),
        # ── NemotronH (layers: attention, mamba, moe, attention) ──
        HeteroCase(
            name="nemotron_h_dim",
            model_key="nemotron_h",
            config_factory=tiny_nemotron_h_config,
            model_cls=NemotronHForCausalLM,
            per_layer_config={0: {"num_key_value_heads": 2}},
        ),
        HeteroCase(
            name="nemotron_h_skip_attn",
            model_key="nemotron_h",
            config_factory=tiny_nemotron_h_config,
            model_cls=NemotronHForCausalLM,
            per_layer_config={3: {"skip": ["mixer"]}},
        ),
        HeteroCase(
            name="nemotron_h_skip_mamba",
            model_key="nemotron_h",
            config_factory=tiny_nemotron_h_config,
            model_cls=NemotronHForCausalLM,
            per_layer_config={1: {"skip": ["mixer"]}},
        ),
        HeteroCase(
            name="nemotron_h_skip_moe",
            model_key="nemotron_h",
            config_factory=tiny_nemotron_h_config,
            model_cls=NemotronHForCausalLM,
            per_layer_config={2: {"skip": ["mixer"]}},
        ),
    ]


def _case_name(f, n, p):
    return f"{f.__name__}_{p.args[0].name}"


def _build_reference(hetero_model, base_config, model_cls, model_key):
    """Build a reference model with skip-aware layers that match the heterogeneous one.

    The reference layers read per-layer config natively: they create modules at the right
    dimensions and simply don't create skipped modules. This is what the model would look
    like if skip support was built in natively rather than through the heterogeneity mechanism.
    """
    hetero_config = hetero_model.config
    ref = build_model(base_config, model_cls)
    ref_layer_cls = MODEL_FIXTURES[model_key].ref_layer_cls
    for i in range(len(ref.model.layers)):
        ref.model.layers[i] = ref_layer_cls(hetero_config, layer_idx=i)
    # Raises on a missing, unexpected, or differently shaped weight, so this also checks every layer's weight shapes
    ref.load_state_dict(hetero_model.state_dict())
    return ref.eval()


def _build_hetero_and_ref(case):
    """Build a heterogeneous model (code under test) and a matching reference model (ground truth).

    The reference uses skip-aware layer subclasses that implement skip/dim natively,
    representing what the model would look like without the generic heterogeneity mechanism.
    """
    hetero = build_model(case.config_factory(per_layer_config=case.per_layer_config), case.model_cls)

    ref = _build_reference(hetero, case.config_factory(), case.model_cls, case.model_key)
    return hetero, ref


@require_torch
class TestSupportedHeterogeneousModels(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @parameterized.expand(HETERO_CASES, name_func=_case_name)
    def test_structure(self, case):
        """The skip replacements match the reference layers."""
        config = case.config_factory(per_layer_config=case.per_layer_config)
        model = build_model(config, case.model_cls)
        modeling_specs = get_heterogeneous_modeling_specs(model)
        self.assertTrue(modeling_specs)

        # Check skip structure: compare hetero model against reference layer expectations.
        replacement_types = tuple(
            type(replacement_factory())
            for modeling_spec in modeling_specs
            for skip_descriptor in modeling_spec.skip_descriptors.values()
            for replacement_factory in skip_descriptor.values()
        )
        ref_layer_cls = MODEL_FIXTURES[case.model_key].ref_layer_cls
        for i in range(config.num_hidden_layers):
            ref_layer = ref_layer_cls(config, layer_idx=i)
            hetero_layer = model.model.layers[i]
            for name, module in hetero_layer.named_children():
                is_replacement = isinstance(module, replacement_types)
                if getattr(ref_layer, name) is None:
                    self.assertTrue(is_replacement, f"Layer {i}.{name} should be a skip replacement")
                else:
                    self.assertFalse(is_replacement, f"Layer {i}.{name} should not be a skip replacement")

    @parameterized.expand(HETERO_CASES, name_func=_case_name)
    def test_forward(self, case):
        """Heterogeneous model forward should match a manually-constructed reference."""
        hetero, ref = _build_hetero_and_ref(case)
        input_ids = dummy_input_ids()
        torch.testing.assert_close(forward_logits(hetero, input_ids), forward_logits(ref, input_ids))

    @parameterized.expand(HETERO_CASES, name_func=_case_name)
    def test_generate(self, case):
        """Heterogeneous model generate should match a manually-constructed reference."""
        hetero, ref = _build_hetero_and_ref(case)
        input_ids = dummy_input_ids()
        gen_kwargs = {"max_new_tokens": 4, "do_sample": False}
        actual_ids = hetero.generate(input_ids, **gen_kwargs)
        # Use an uncached reference so a shared cache bug cannot make both outputs agree.
        expected_ids = ref.generate(input_ids, use_cache=False, **gen_kwargs)
        torch.testing.assert_close(actual_ids, expected_ids)
