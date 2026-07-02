# Copyright 2026 The HuggingFace Inc. team.
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

import copy
import unittest

from transformers import is_torch_available
from transformers.module_fusion import FusedModule, ModuleSpec, RegistryCollector, fuse_modules, unfuse_modules
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch
    import torch.nn as nn


class LinearWithScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x, scale=1.0):
        return self.linear(x) * scale


class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8)

    def forward(self, x, bias=None):
        x = self.norm(x)
        if bias is not None:
            x = x + bias
        return x


class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = LinearWithScale()
        self.norm = LayerNorm()

    def forward(self, x, scale=1.0):
        x = self.linear(x, scale=scale)
        x = self.norm(x)
        return x


class DummyModel(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([DummyBlock() for _ in range(num_layers)])

    def forward(self, x, scale=1.0):
        for block in self.layers:
            x = block(x, scale=scale)
        return x


@require_torch
class TestModuleFusion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(2, 8)

    # --- RegistryCollector ---

    def test_collector_captures_input_and_is_passthrough(self):
        """RegistryCollector stores args in registry and returns input unchanged."""
        registry = {}
        spec = ModuleSpec(inputs=["x"], outputs=["x"])
        linear = LinearWithScale()
        collector = RegistryCollector(spec, index=2, registry=registry, orig_module=linear)
        x = torch.randn(2, 8)
        out = collector(x)
        self.assertIn("in_2_x", registry)
        self.assertIs(registry["in_2_x"], x)
        self.assertTrue(torch.equal(out, x))

    def test_collector_delegates_attribute_access_to_orig_module(self):
        """Attribute access on RegistryCollector is transparently forwarded to orig_module."""
        linear = LinearWithScale()
        collector = RegistryCollector(
            ModuleSpec(inputs=["x", "scale"], outputs=["x"]), index=0, registry={}, orig_module=linear
        )
        self.assertIs(collector.linear, linear.linear)
        self.assertIs(collector.linear.weight, linear.linear.weight)

    # --- FusedModule validation ---

    def test_fused_module_raises_on_spec_count_mismatch(self):
        """Mismatched number of modules vs specs raises ValueError."""
        with self.assertRaises(ValueError):
            FusedModule(
                [LinearWithScale(), LayerNorm()],
                [ModuleSpec(inputs=["x", "scale"], outputs=["x"])],  # only 1 spec for 2 modules
                {},
            )

    def test_fused_module_raises_on_input_count_mismatch(self):
        """Spec with more inputs than module params raises ValueError."""
        with self.assertRaises(ValueError):
            FusedModule(
                [LinearWithScale()],  # forward(x, scale) → 2 params total
                [ModuleSpec(inputs=["x", "scale", "extra"], outputs=["x"])],  # 3 > 2 params
                {},
            )

    def test_fused_module_delegates_attribute_access_to_last_module(self):
        """Attribute access on FusedModule is transparently forwarded to the last module in the chain."""
        linear = LinearWithScale()
        norm = LayerNorm()
        specs = [
            ModuleSpec(inputs=["x", "scale"], outputs=["x"]),
            ModuleSpec(inputs=["x"], outputs=["x"]),
        ]
        fused = FusedModule([linear, norm], specs, {})
        self.assertIs(fused.norm, norm.norm)
        self.assertIs(fused.norm.weight, norm.norm.weight)

    # --- FusedModule forward ---

    def test_fused_module_chains_outputs_to_next_inputs(self):
        """Output of module 0 is passed as input to module 1 via registry."""
        linear = LinearWithScale()
        norm = LayerNorm()
        specs = [
            ModuleSpec(inputs=["x", "scale"], outputs=["x"]),
            ModuleSpec(inputs=["x", "bias"], outputs=["x"]),
        ]
        # bias is not produced by linear, must be pre-populated in the registry
        registry = {"in_1_bias": None}
        fused = FusedModule([linear, norm], specs, registry)
        out = fused(self.x, torch.tensor(1.0))
        expected = norm(linear(self.x, scale=1.0))
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_fused_module_fallback_for_external_inputs(self):
        """Input not produced by prior module falls back to collector-captured in_{i}_{name}."""
        linear = LinearWithScale()
        norm = LayerNorm()
        bias = torch.ones(8) * 0.5
        specs = [
            ModuleSpec(inputs=["x", "scale"], outputs=["x"]),
            ModuleSpec(inputs=["x", "bias"], outputs=["x"]),
        ]
        registry = {"in_1_bias": bias}
        fused = FusedModule([linear, norm], specs, registry)
        out = fused(self.x, torch.tensor(1.0))
        expected = norm(linear(self.x, scale=1.0), bias=bias)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_fused_module_registry_cleared_after_forward(self):
        """Registry is empty after FusedModule.forward() so no state leaks between calls."""
        linear = LinearWithScale()
        registry = {}
        fused = FusedModule([linear], [ModuleSpec(inputs=["x", "scale"], outputs=["x"])], registry)
        fused(self.x, torch.tensor(1.0))
        self.assertEqual(len(registry), 0)

    # --- fuse_modules ---

    def test_fuse_modules_structure(self):
        """fuse_modules places RegistryCollectors on all but last, FusedModule on last."""
        model = DummyModel(num_layers=1)
        specs = [
            ModuleSpec(inputs=["x", "scale"], outputs=["x"]),
            ModuleSpec(inputs=["x"], outputs=["x"]),  # bias has a default, omitted from spec
        ]
        fuse_modules(model, ["layers.*.linear", "layers.*.norm"], specs)
        self.assertIsInstance(model.layers[0].linear, RegistryCollector)
        self.assertIsInstance(model.layers[0].norm, FusedModule)

    def test_fuse_modules_numerical_equivalence(self):
        """Fused model produces identical output to original for all layers."""
        model = DummyModel(num_layers=3)
        original = copy.deepcopy(model)
        specs = [
            ModuleSpec(inputs=["x", "scale"], outputs=["x"]),
            ModuleSpec(inputs=["x"], outputs=["x"]),  # bias has a default, omitted from spec
        ]
        fuse_modules(model, ["layers.*.linear", "layers.*.norm"], specs)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(self.x), original(self.x), atol=1e-6))

    def test_fuse_modules_each_layer_has_independent_registry(self):
        """Each fused group uses its own registry; collector and FusedModule share it."""
        specs = [
            ModuleSpec(inputs=["x", "scale"], outputs=["x"]),
            ModuleSpec(inputs=["x"], outputs=["x"]),
        ]
        num_layers = 5
        model = DummyModel(num_layers=num_layers)
        fuse_modules(model, ["layers.*.linear", "layers.*.norm"], specs)
        registries = []
        for index in range(num_layers):
            block = model.layers[index]
            registries.append(block.linear._registry)

        # All layers should have distinct registry objects
        registries_ids = {id(reg) for reg in registries}
        self.assertEqual(len(registries_ids), num_layers)

    # --- unfuse_modules ---

    def test_unfuse_restores_modules_and_numerical_equivalence(self):
        """After fuse+unfuse, original modules are restored and output matches original."""
        model = DummyModel(num_layers=2)
        original = copy.deepcopy(model)
        orig_linear = model.layers[0].linear
        specs = [
            ModuleSpec(inputs=["x", "scale"], outputs=["x"]),
            ModuleSpec(inputs=["x"], outputs=["x"]),
        ]
        fuse_modules(model, ["layers.*.linear", "layers.*.norm"], specs)
        unfuse_modules(model)
        self.assertIs(model.layers[0].linear, orig_linear)
        self.assertNotIsInstance(model.layers[0].norm, FusedModule)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(self.x), original(self.x), atol=1e-6))
