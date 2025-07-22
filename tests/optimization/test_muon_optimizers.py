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


import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch
    from torch import nn

    from transformers.optimization import (
        MuonWithAuxAdam,
        SingleDeviceMuon,
        SingleDeviceMuonWithAuxAdam,
        adam_update,
    )


@require_torch
class MuonOptimizersTest(unittest.TestCase):
    def assertListAlmostEqual(self, list1, list2, tol):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, delta=tol)

    def test_adam_update_helper(self):
        """Test the adam_update helper function"""
        grad = torch.tensor([0.1, -0.2, 0.3], requires_grad=False)
        buf1 = torch.zeros_like(grad)
        buf2 = torch.zeros_like(grad)
        step = 1
        betas = (0.9, 0.95)
        eps = 1e-8

        update = adam_update(grad, buf1, buf2, step, betas, eps)

        # Check that update has the correct shape
        self.assertEqual(update.shape, grad.shape)

        # Check that buffers were updated
        self.assertFalse(torch.allclose(buf1, torch.zeros_like(grad)))
        self.assertFalse(torch.allclose(buf2, torch.zeros_like(grad)))

    def test_single_device_muon_initialization(self):
        """Test SingleDeviceMuon optimizer initialization"""
        model = nn.Linear(10, 5)
        optimizer = SingleDeviceMuon(model.parameters(), lr=0.02, weight_decay=0.01, momentum=0.95)

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.02)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.01)
        self.assertEqual(optimizer.param_groups[0]["momentum"], 0.95)

    def test_single_device_muon_step(self):
        """Test SingleDeviceMuon optimizer step function"""
        w = torch.tensor([[0.1, -0.2], [-0.1, 0.3]], requires_grad=True)
        target = torch.tensor([[0.4, 0.2], [-0.5, 0.1]])
        criterion = nn.MSELoss()

        optimizer = SingleDeviceMuon([w], lr=0.1, weight_decay=0.0, momentum=0.95)

        initial_w = w.clone()
        for _ in range(10):
            loss = criterion(w, target)
            loss.backward()
            optimizer.step()
            w.grad.detach_()
            w.grad.zero_()

        # Check that weights were updated
        self.assertFalse(torch.allclose(w, initial_w))

    def test_muon_with_aux_adam_initialization(self):
        """Test MuonWithAuxAdam optimizer initialization"""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),  # This will be optimized with Muon
            nn.Linear(20, 5),  # This will be optimized with Adam
        )

        # Separate parameters as in the provided example
        hidden_weights = [p for p in model[0].parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model[0].parameters() if p.ndim < 2]
        nonhidden_params = list(model[1].parameters())

        param_groups = [
            {"params": hidden_weights, "use_muon": True, "lr": 0.02, "weight_decay": 0.01, "momentum": 0.95},
            {
                "params": hidden_gains_biases + nonhidden_params,
                "use_muon": False,
                "lr": 3e-4,
                "betas": (0.9, 0.95),
                "weight_decay": 0.01,
                "eps": 1e-8,
            },
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        self.assertEqual(len(optimizer.param_groups), 2)

        # Check Muon group
        muon_group = optimizer.param_groups[0]
        self.assertTrue(muon_group["use_muon"])
        self.assertEqual(muon_group["lr"], 0.02)
        self.assertEqual(muon_group["weight_decay"], 0.01)
        self.assertEqual(muon_group["momentum"], 0.95)

        # Check Adam group
        adam_group = optimizer.param_groups[1]
        self.assertFalse(adam_group["use_muon"])
        self.assertEqual(adam_group["lr"], 3e-4)
        self.assertEqual(adam_group["betas"], (0.9, 0.95))
        self.assertEqual(adam_group["weight_decay"], 0.01)
        self.assertEqual(adam_group["eps"], 1e-8)

    def test_single_device_muon_with_aux_adam_initialization(self):
        """Test SingleDeviceMuonWithAuxAdam optimizer initialization"""

        # Create a simple model similar to the provided example
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 20))
                self.head = nn.Linear(20, 5)
                self.embed = nn.Embedding(100, 10)

            def forward(self, x):
                x = self.body(x)
                return self.head(x)

        model = SimpleModel()

        # Separate parameters as in the provided example
        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
        nonhidden_params = [*model.head.parameters(), *model.embed.parameters()]

        param_groups = [
            {"params": hidden_weights, "use_muon": True, "lr": 0.02, "weight_decay": 0.01, "momentum": 0.95},
            {
                "params": hidden_gains_biases + nonhidden_params,
                "use_muon": False,
                "lr": 3e-4,
                "betas": (0.9, 0.95),
                "weight_decay": 0.01,
                "eps": 1e-8,
            },
        ]

        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        self.assertEqual(len(optimizer.param_groups), 2)

        # Check Muon group
        muon_group = optimizer.param_groups[0]
        self.assertTrue(muon_group["use_muon"])
        self.assertEqual(muon_group["lr"], 0.02)
        self.assertEqual(muon_group["weight_decay"], 0.01)
        self.assertEqual(muon_group["momentum"], 0.95)

        # Check Adam group
        adam_group = optimizer.param_groups[1]
        self.assertFalse(adam_group["use_muon"])
        self.assertEqual(adam_group["lr"], 3e-4)
        self.assertEqual(adam_group["betas"], (0.9, 0.95))
        self.assertEqual(adam_group["weight_decay"], 0.01)
        self.assertEqual(adam_group["eps"], 1e-8)

    def test_single_device_muon_with_aux_adam_step(self):
        """Test SingleDeviceMuonWithAuxAdam optimizer step function"""

        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 10))
                self.head = nn.Linear(10, 3)

            def forward(self, x):
                x = self.body(x)
                return self.head(x)

        model = SimpleModel()

        # Separate parameters
        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
        nonhidden_params = list(model.head.parameters())

        param_groups = [
            {"params": hidden_weights, "use_muon": True, "lr": 0.02, "weight_decay": 0.01, "momentum": 0.95},
            {
                "params": hidden_gains_biases + nonhidden_params,
                "use_muon": False,
                "lr": 3e-4,
                "betas": (0.9, 0.95),
                "weight_decay": 0.01,
                "eps": 1e-8,
            },
        ]

        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        # Create some dummy data
        x = torch.randn(32, 5)
        target = torch.randn(32, 3)
        criterion = nn.MSELoss()

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Run a few optimization steps
        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Check that at least some parameters were updated
        updated = False
        for initial, current in zip(initial_params, model.parameters()):
            if not torch.allclose(initial, current, atol=1e-6):
                updated = True
                break

        self.assertTrue(updated, "Parameters should have been updated after optimization steps")

    def test_muon_with_aux_adam_param_group_validation(self):
        """Test that MuonWithAuxAdam validates param_groups correctly"""
        w1 = torch.tensor([[0.1, -0.2]], requires_grad=True)
        w2 = torch.tensor([0.3], requires_grad=True)

        # Test missing use_muon flag
        with self.assertRaises(AssertionError):
            param_groups = [{"params": [w1], "lr": 0.02}]
            MuonWithAuxAdam(param_groups)

        # Test invalid keys for Muon group
        with self.assertRaises(AssertionError):
            param_groups = [{"params": [w1], "use_muon": True, "lr": 0.02, "invalid_key": True}]
            MuonWithAuxAdam(param_groups)

        # Test invalid keys for Adam group
        with self.assertRaises(AssertionError):
            param_groups = [{"params": [w2], "use_muon": False, "lr": 3e-4, "invalid_key": True}]
            MuonWithAuxAdam(param_groups)

    def test_single_device_muon_with_aux_adam_param_group_validation(self):
        """Test that SingleDeviceMuonWithAuxAdam validates param_groups correctly"""
        w1 = torch.tensor([[0.1, -0.2]], requires_grad=True)
        w2 = torch.tensor([0.3], requires_grad=True)

        # Test missing use_muon flag
        with self.assertRaises(AssertionError):
            param_groups = [{"params": [w1], "lr": 0.02}]
            SingleDeviceMuonWithAuxAdam(param_groups)

        # Test invalid keys for Muon group
        with self.assertRaises(AssertionError):
            param_groups = [{"params": [w1], "use_muon": True, "lr": 0.02, "invalid_key": True}]
            SingleDeviceMuonWithAuxAdam(param_groups)

        # Test invalid keys for Adam group
        with self.assertRaises(AssertionError):
            param_groups = [{"params": [w2], "use_muon": False, "lr": 3e-4, "invalid_key": True}]
            SingleDeviceMuonWithAuxAdam(param_groups)

    def test_muon_default_values(self):
        """Test that default parameter values are set correctly"""
        w1 = torch.tensor([[0.1, -0.2]], requires_grad=True)
        w2 = torch.tensor([0.3], requires_grad=True)

        # Test defaults for SingleDeviceMuonWithAuxAdam
        param_groups = [
            {"params": [w1], "use_muon": True},
            {"params": [w2], "use_muon": False},
        ]

        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        # Check Muon defaults
        muon_group = optimizer.param_groups[0]
        self.assertEqual(muon_group["lr"], 0.02)
        self.assertEqual(muon_group["momentum"], 0.95)
        self.assertEqual(muon_group["weight_decay"], 0)

        # Check Adam defaults
        adam_group = optimizer.param_groups[1]
        self.assertEqual(adam_group["lr"], 3e-4)
        self.assertEqual(adam_group["betas"], (0.9, 0.95))
        self.assertEqual(adam_group["eps"], 1e-10)
        self.assertEqual(adam_group["weight_decay"], 0)

    def test_inference_example_usage(self):
        """Test the exact usage pattern from the provided inference example"""

        # Create a model structure similar to the provided example
        class ExampleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.LayerNorm(20),  # This creates 1D parameters (gains/biases)
                    nn.Linear(20, 15),
                    nn.LayerNorm(15),
                )
                self.head = nn.Linear(15, 5)
                self.embed = nn.Embedding(100, 10)

            def forward(self, x):
                # For embedding, we need token indices, not continuous values
                # So we'll just use the body and head for this test
                x = self.body(x)
                return self.head(x)

        model = ExampleModel()

        # Use the exact parameter separation from the provided example
        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
        nonhidden_params = [*model.head.parameters(), *model.embed.parameters()]

        param_groups = [
            {"params": hidden_weights, "use_muon": True, "lr": 0.02, "weight_decay": 0.01},
            {
                "params": hidden_gains_biases + nonhidden_params,
                "use_muon": False,
                "lr": 3e-4,
                "betas": (0.9, 0.95),
                "weight_decay": 0.01,
            },
        ]

        # Test that the optimizer can be created and used
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        # Verify the parameter groups are set up correctly
        self.assertEqual(len(optimizer.param_groups), 2)

        # Check that we have the expected number of parameters in each group
        muon_group = optimizer.param_groups[0]
        adam_group = optimizer.param_groups[1]

        # Should have 2D parameters (weight matrices) in Muon group
        self.assertTrue(all(p.ndim >= 2 for p in muon_group["params"]))

        # Adam group should contain 1D parameters and all head/embed parameters
        self.assertTrue(len(adam_group["params"]) > 0)

        # Test that optimizer can perform steps
        x = torch.randn(16, 10)
        target = torch.randn(16, 5)
        criterion = nn.MSELoss()

        for _ in range(3):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # If we get here without errors, the test passes
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
