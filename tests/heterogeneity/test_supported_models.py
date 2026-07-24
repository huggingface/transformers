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

import tempfile
import unittest
from dataclasses import dataclass, field
from typing import NamedTuple

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
        hetero_context,
        tiny_gpt_oss_config,
        tiny_llama4_config,
        tiny_llama_config,
        tiny_nemotron_h_config,
    )
    from transformers import LlamaForCausalLM
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
    from transformers.models.llama4.modeling_llama4 import Llama4ForCausalLM
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

    class WeightCheck(NamedTuple):
        """An expected weight shape in the built model."""

        layer_idx: int
        attr_path: str  # dot-separated path to the weight tensor
        shape_dim: int
        expected: int

    @dataclass
    class HeteroCase:
        """A single heterogeneous modeling test scenario.

        Fields describe the model under test AND how to build a manually-constructed
        reference model (ground truth) that does not use any heterogeneity code.
        """

        name: str
        model_key: str  # key into MODEL_FIXTURES
        config_factory: callable
        model_cls: type
        per_layer_config: dict
        # --- Structure verification fields ---
        structure_weight_checks: list[WeightCheck] = field(default_factory=list)

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
            structure_weight_checks=[
                WeightCheck(0, "mlp.gate_proj.weight", 0, 64),
                WeightCheck(0, "self_attn.k_proj.weight", 0, 32),  # 2 kv_heads × 16
                WeightCheck(1, "mlp.gate_proj.weight", 0, 128),
                WeightCheck(1, "self_attn.k_proj.weight", 0, 64),  # 4 kv_heads × 16 (default)
                WeightCheck(2, "mlp.gate_proj.weight", 0, 96),
                WeightCheck(2, "self_attn.k_proj.weight", 0, 16),  # 1 kv_head × 16
                WeightCheck(3, "mlp.gate_proj.weight", 0, 128),
                WeightCheck(3, "self_attn.k_proj.weight", 0, 64),  # 4 kv_heads × 16 (default)
            ],
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
            structure_weight_checks=[
                WeightCheck(0, "mlp.experts.down_proj", 1, 16),
                WeightCheck(1, "mlp.experts.down_proj", 1, 32),
                WeightCheck(2, "mlp.experts.down_proj", 1, 48),
                WeightCheck(3, "mlp.experts.down_proj", 1, 32),
            ],
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
        HeteroCase(
            name="gpt_oss_skip_both",
            model_key="gpt_oss",
            config_factory=tiny_gpt_oss_config,
            model_cls=GptOssForCausalLM,
            per_layer_config={1: {"skip": ["attention", "mlp"]}},
        ),
        # ── Llama4 ──
        HeteroCase(
            name="llama4_dim",
            model_key="llama4",
            config_factory=tiny_llama4_config,
            model_cls=Llama4ForCausalLM,
            per_layer_config={0: {"intermediate_size_mlp": 64}},
            structure_weight_checks=[
                WeightCheck(0, "feed_forward.up_proj.weight", 0, 64),
                WeightCheck(2, "feed_forward.up_proj.weight", 0, 128),  # default
            ],
        ),
        HeteroCase(
            name="llama4_skip_attn",
            model_key="llama4",
            config_factory=tiny_llama4_config,
            model_cls=Llama4ForCausalLM,
            per_layer_config={0: {"skip": ["attention"]}},
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
        HeteroCase(
            name="llama4_skip_both",
            model_key="llama4",
            config_factory=tiny_llama4_config,
            model_cls=Llama4ForCausalLM,
            per_layer_config={0: {"skip": ["attention", "mlp"]}},
        ),
        HeteroCase(
            name="llama4_skip_moe_both",
            model_key="llama4",
            config_factory=tiny_llama4_config,
            model_cls=Llama4ForCausalLM,
            per_layer_config={1: {"skip": ["attention", "mlp"]}},
        ),
        # ── NemotronH (layers: attention, mamba, moe, attention) ──
        HeteroCase(
            name="nemotron_h_dim",
            model_key="nemotron_h",
            config_factory=tiny_nemotron_h_config,
            model_cls=NemotronHForCausalLM,
            per_layer_config={0: {"num_key_value_heads": 2}},
            structure_weight_checks=[
                WeightCheck(0, "mixer.k_proj.weight", 0, 32),  # 2 kv_heads × head_dim 16
                WeightCheck(3, "mixer.k_proj.weight", 0, 64),  # 4 kv_heads × 16 (default)
            ],
        ),
        HeteroCase(
            name="nemotron_h_skip_attn",
            model_key="nemotron_h",
            config_factory=tiny_nemotron_h_config,
            model_cls=NemotronHForCausalLM,
            per_layer_config={0: {"skip": ["mixer"]}},
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
    ref.load_state_dict(hetero_model.state_dict())
    return ref.eval()


def _build_hetero_and_ref(case):
    """Build a heterogeneous model (code under test) and a matching reference model (ground truth).

    The reference uses skip-aware layer subclasses that implement skip/dim natively,
    representing what the model would look like without the generic heterogeneity mechanism.
    """
    with hetero_context(case.model_key):
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
        """Verify the entire model structure: skip replacements and weight shapes."""
        config = case.config_factory(per_layer_config=case.per_layer_config)
        with hetero_context(case.model_key) as modeling_spec:
            model = build_model(config, case.model_cls)

        # Check skip structure: compare hetero model against reference layer expectations.
        replacement_types = tuple(
            type(replacement_factory())
            for skip_descriptor in (modeling_spec.skip_descriptors or {}).values()
            for replacement_factory in skip_descriptor.replacements.values()
        )
        expected_disabled_kv_layer_indices = tuple(
            layer_idx
            for layer_idx, layer_overrides in sorted(case.per_layer_config.items())
            if any(
                modeling_spec.skip_descriptors[skip_type].replaces_kv_cache_updater
                for skip_type in layer_overrides.get("skip", ())
            )
        )
        self.assertEqual(
            model.config._heterogeneity_spec.disabled_kv_layer_indices, expected_disabled_kv_layer_indices
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

        # Check weight shapes on all specified layers
        for layer_idx, attr_path, shape_dim, expected in case.structure_weight_checks:
            obj = model.model.layers[layer_idx]
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            self.assertEqual(obj.shape[shape_dim], expected, f"Layer {layer_idx} {attr_path} shape mismatch")

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
        self.assertTrue(
            torch.equal(
                hetero.generate(input_ids, **gen_kwargs),
                ref.generate(input_ids, **gen_kwargs),
            )
        )

    def test_save_pretrained_model_round_trip(self):
        """Full model save/load: config, weight shapes, and forward output should survive."""
        per_layer = {0: {"intermediate_size": 64}, 2: {"intermediate_size": 96}}
        hetero_config = tiny_llama_config(per_layer_config=per_layer)
        with hetero_context("llama"):
            hetero_model = build_model(hetero_config, LlamaForCausalLM)

        input_ids = dummy_input_ids()
        expected_logits = forward_logits(hetero_model, input_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            hetero_model.save_pretrained(tmpdir)
            with hetero_context("llama"):
                loaded_model = LlamaForCausalLM.from_pretrained(tmpdir)

        loaded_model.eval()
        for layer_idx in range(4):
            orig_shape = hetero_model.model.layers[layer_idx].mlp.gate_proj.weight.shape
            loaded_shape = loaded_model.model.layers[layer_idx].mlp.gate_proj.weight.shape
            self.assertEqual(orig_shape, loaded_shape, f"Layer {layer_idx} weight shape mismatch")

        torch.testing.assert_close(forward_logits(loaded_model, input_ids), expected_logits)
