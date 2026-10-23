# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
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
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    import torch
    from torch import nn

    from transformers import SST, Adafactor, get_linear_schedule_with_warmup

    class _Model(nn.Module):
        def __init__(self, dtype=torch.float32):
            super().__init__()
            self.q_proj = nn.Linear(6, 8, dtype=dtype)
            self.other = nn.Linear(6, 8, bias=False, dtype=dtype)


@require_torch
class SSTTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = _Model().to(torch_device)

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_state_equal(first, second)
        else:
            self.assertEqual(left, right)

    def assert_protected(self, optimizer, tolerance=3e-6):
        parameters = dict(self.model.named_parameters())
        for item in optimizer.sst_state_dict()["protected_weights"]:
            matrix = parameters[item["name"]].detach()
            left, right = item["left"].to(matrix.device), item["right"].to(matrix.device)
            anchor = (left * item["singular_values"].to(matrix.device).unsqueeze(0)) @ right.mH
            delta = matrix - anchor
            self.assertLess((left.mH @ delta).abs().max().item(), tolerance)
            self.assertLess((delta @ right).abs().max().item(), tolerance)

    @parameterized.expand(
        [
            (name, refresh_every)
            for name in (
                "SGD",
                "Adam",
                "AdamW",
                "RMSprop",
                "Adagrad",
                "Adadelta",
                "Adamax",
                "ASGD",
                "Rprop",
                "NAdam",
                "RAdam",
                "Adafactor",
                "TorchAdafactor",
                "Muon",
            )
            for refresh_every in (None, 2)
        ]
    )
    def test_wraps_optimizer_without_changing_its_state(self, name, refresh_every):
        if name == "Adafactor":
            inner = Adafactor(self.model.parameters(), lr=0.05, relative_step=False, scale_parameter=False)
        elif name == "TorchAdafactor":
            if not hasattr(torch.optim, "Adafactor"):
                self.skipTest("PyTorch Adafactor is not available in this version.")
            inner = torch.optim.Adafactor(self.model.parameters(), lr=0.05)
        elif name == "Muon":
            if not hasattr(torch.optim, "Muon"):
                self.skipTest("PyTorch Muon is not available in this version.")
            inner = torch.optim.Muon([self.model.q_proj.weight], lr=0.05)
        else:
            inner = getattr(torch.optim, name)(self.model.parameters(), lr=0.05)
        optimizer = SST(
            inner, model=self.model, rank_fraction=0.5, target_modules=["q_proj"], refresh_every=refresh_every
        )
        raw_states, raw_weights = [], []

        def after_inner_step(*_):
            raw_states.append(copy.deepcopy(inner.state_dict()))
            raw_weights.append({name: parameter.detach().clone() for name, parameter in self.model.named_parameters()})

        handle = inner.register_step_post_hook(after_inner_step)
        for _ in range(3):
            gradients = {}
            for parameter_name, parameter in self.model.named_parameters():
                parameter.grad = torch.randn_like(parameter)
                gradients[parameter_name] = parameter.grad.clone()
            optimizer.step()
            self.assert_state_equal(raw_states[-1], inner.state_dict())
            self.assert_protected(optimizer)
            for parameter_name, parameter in self.model.named_parameters():
                torch.testing.assert_close(parameter.grad, gradients[parameter_name], rtol=0, atol=0)
                if parameter_name != "q_proj.weight":
                    torch.testing.assert_close(parameter, raw_weights[-1][parameter_name], rtol=0, atol=0)
            self.assertIs(optimizer.param_groups, inner.param_groups)
            self.assertIs(optimizer.state, inner.state)
        handle.remove()
        self.assertEqual(optimizer.num_updates, 3)
        self.assertEqual(optimizer.num_refreshes, 0 if refresh_every is None else 1)
        self.assertFalse(torch.equal(self.model.q_proj.weight, raw_weights[-1]["q_proj.weight"]))

    def test_projection_matches_projected_applied_update(self):
        anchor = self.model.q_proj.weight.detach().clone()
        inner = torch.optim.SGD(self.model.parameters(), lr=0.1, weight_decay=0.2, momentum=0.9)
        optimizer = SST(inner, model=self.model, rank_fraction=0.5, target_modules=["q_proj"])
        raw_weight = []
        inner.register_step_post_hook(lambda *_: raw_weight.append(self.model.q_proj.weight.detach().clone()))
        self.model.q_proj.weight.grad = torch.randn_like(anchor)
        optimizer.step()
        item = optimizer.sst_state_dict()["protected_weights"][0]
        left, right = item["left"].to(anchor.device), item["right"].to(anchor.device)
        delta = raw_weight[0] - anchor
        delta = delta - left @ (left.mH @ delta)
        delta = delta - (delta @ right) @ right.mH
        torch.testing.assert_close(self.model.q_proj.weight, anchor + delta, rtol=2e-6, atol=2e-6)

    def test_without_model_selects_only_owned_matrices(self):
        inner = torch.optim.SGD(self.model.q_proj.parameters(), lr=0.1)
        optimizer = SST(inner, rank_fraction=0.5)
        self.assertEqual([item["name"] for item in optimizer.sst_state_dict()["protected_weights"]], ["parameter_0"])
        self.model.q_proj.weight.grad = torch.randn_like(self.model.q_proj.weight)
        optimizer.step()

    def test_optimizer_subset_is_respected(self):
        optimizer = SST(torch.optim.AdamW(self.model.q_proj.parameters()), model=self.model, rank_fraction=0.5)
        self.assertEqual([item["name"] for item in optimizer.sst_state_dict()["protected_weights"]], ["q_proj.weight"])

    @parameterized.expand(["bit16_groups", "fp16_groups"])
    def test_zero_ownership_does_not_depend_on_local_fragments(self, attribute):
        master_partition = nn.Parameter(torch.zeros(8, device=torch_device))
        inner = torch.optim.SGD([master_partition], lr=0.1)
        setattr(inner, attribute, [[self.model.q_proj.weight]])
        self.model.q_proj.weight._hp_mapping = None
        self.model.other.weight._z3_optimizer = Mock()
        optimizer = SST(inner, model=self.model, initialize=False)
        self.assertEqual([name for name, _ in optimizer._parameters()], ["q_proj.weight"])

    @parameterized.expand([(None,), (1,)])
    def test_sparse_optimizer_and_gradient_are_preserved(self, refresh_every):
        self.model = nn.Sequential(nn.Embedding(12, 8, sparse=True)).to(torch_device)
        # Match the linear fixtures' scale for the same absolute fp32 projection tolerance.
        with torch.no_grad():
            self.model[0].weight.mul_(0.1)
        inner = torch.optim.SparseAdam(self.model.parameters(), lr=0.05)
        optimizer = SST(inner, model=self.model, rank_fraction=0.5, refresh_every=refresh_every)
        self.model(torch.tensor([1, 3, 5], device=torch_device)).sum().backward()
        gradient = self.model[0].weight.grad.clone()
        before_state = []
        inner.register_step_post_hook(lambda *_: before_state.append(copy.deepcopy(inner.state_dict())))
        optimizer.step()
        self.assert_state_equal(before_state[0], inner.state_dict())
        torch.testing.assert_close(self.model[0].weight.grad, gradient, rtol=0, atol=0)
        self.assert_protected(optimizer)

    @parameterized.expand([(None,), ("all-linear",), ([r".*"],), (["q_proj", "other"],)])
    def test_target_selection(self, targets):
        optimizer = SST(
            torch.optim.SGD(self.model.parameters(), lr=0.1),
            model=self.model,
            target_modules=targets,
            rank_fraction=0.5,
        )
        self.assertEqual(
            {item["name"] for item in optimizer.sst_state_dict()["protected_weights"]},
            {"q_proj.weight", "other.weight"},
        )

    def test_parameter_groups_and_scheduler_are_shared(self):
        inner = torch.optim.SGD(
            [
                {"params": self.model.q_proj.parameters(), "lr": 0.1},
                {"params": self.model.other.parameters(), "lr": 0.02},
            ]
        )
        optimizer = SST(inner, model=self.model, rank_fraction=0.5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=4)
        for _ in range(2):
            self.model.q_proj.weight.grad = torch.randn_like(self.model.q_proj.weight)
            optimizer.step()
            scheduler.step()
        self.assertEqual([group["lr"] for group in inner.param_groups], [0.05, 0.01])
        self.assertIs(scheduler.optimizer, optimizer)
        optimizer.zero_grad()
        self.assertIsNone(self.model.q_proj.weight.grad)

    def test_add_parameter_group_preserves_old_basis(self):
        optimizer = SST(torch.optim.AdamW(self.model.parameters()), model=self.model, rank_fraction=0.5)
        original = copy.deepcopy(optimizer.sst_state_dict())
        self.model.extra = nn.Parameter(torch.randn(5, 4, device=torch_device))
        optimizer.add_param_group({"params": [self.model.extra], "lr": 0.02})
        self.assertEqual(len(optimizer.sst_state_dict()["protected_weights"]), 3)
        self.assert_state_equal(original["protected_weights"], optimizer.sst_state_dict()["protected_weights"][:2])
        self.model.extra.grad = torch.randn_like(self.model.extra)
        optimizer.step()
        self.assert_protected(optimizer)

    @parameterized.expand([(None,), (1,)])
    def test_closure_and_return_value(self, refresh_every):
        inner = torch.optim.LBFGS([self.model.q_proj.weight], lr=0.05, max_iter=3)
        optimizer = SST(inner, model=self.model, rank_fraction=0.5, refresh_every=refresh_every)
        calls = []

        def closure():
            optimizer.zero_grad()
            loss = self.model.q_proj.weight.square().sum()
            loss.backward()
            calls.append(loss)
            return loss

        result = optimizer.step(closure)
        self.assertIs(result, calls[0])
        self.assertEqual(optimizer.num_updates, 1)
        self.assertEqual(optimizer.num_refreshes, 0 if refresh_every is None else 1)
        self.assert_protected(optimizer)

    def test_manual_refresh_reanchors_without_modifying_weights_gradients_or_inner_state(self):
        inner = torch.optim.AdamW(self.model.parameters(), lr=0.1)
        optimizer = SST(inner, model=self.model, rank_fraction=0.5)
        for parameter in self.model.parameters():
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        with torch.no_grad():
            self.model.q_proj.weight.add_(0.1 * torch.randn_like(self.model.q_proj.weight))
        before_weights = copy.deepcopy(self.model.state_dict())
        before_gradients = [parameter.grad.clone() for parameter in self.model.parameters()]
        before_inner = copy.deepcopy(inner.state_dict())
        previous = copy.deepcopy(optimizer.sst_state_dict())
        optimizer.refresh()
        self.assert_state_equal(before_weights, self.model.state_dict())
        self.assert_state_equal(before_inner, inner.state_dict())
        for gradient, parameter in zip(before_gradients, self.model.parameters()):
            torch.testing.assert_close(gradient, parameter.grad, rtol=0, atol=0)
        self.assertEqual(optimizer.num_updates, 1)
        self.assertEqual(optimizer.last_refresh_step, 1)
        self.assertEqual(optimizer.num_refreshes, 1)
        self.assertFalse(torch.equal(previous["protected_weights"][0]["left"], optimizer._basis["q_proj.weight"].left))
        self.assert_protected(optimizer)
        with patch("torch.linalg.svd", side_effect=AssertionError("initialize must not refresh existing targets")):
            optimizer.initialize()
        self.assertEqual(optimizer.num_refreshes, 1)
        optimizer.step()
        self.assert_protected(optimizer)

    def test_manual_refresh_restarts_the_periodic_interval(self):
        optimizer = SST(
            torch.optim.SGD(self.model.parameters(), lr=0.1),
            model=self.model,
            rank_fraction=0.5,
            refresh_every=3,
        )
        self.model.q_proj.weight.grad = torch.randn_like(self.model.q_proj.weight)
        optimizer.step()
        optimizer.refresh()
        for step in (2, 3, 4):
            optimizer.step()
            self.assertEqual(optimizer.num_updates, step)
            self.assertEqual(optimizer.num_refreshes, 2 if step == 4 else 1)
        self.assertEqual(optimizer.last_refresh_step, 4)

    def test_refresh_failure_keeps_every_active_basis_and_counter(self):
        optimizer = SST(torch.optim.AdamW(self.model.parameters()), model=self.model, rank_fraction=0.5)
        original = copy.deepcopy(optimizer.sst_state_dict())
        svd = torch.linalg.svd
        calls = []

        def fail_second_target(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("Injected SVD failure")
            return svd(*args, **kwargs)

        with (
            patch("torch.linalg.svd", side_effect=fail_second_target),
            self.assertRaisesRegex(RuntimeError, "active basis was not replaced"),
        ):
            optimizer.refresh()
        self.assert_state_equal(original, optimizer.sst_state_dict())

    def test_uninitialized_refresh_requires_an_explicit_initial_basis(self):
        optimizer = SST(torch.optim.SGD(self.model.parameters(), lr=0.1), model=self.model, initialize=False)
        with self.assertRaisesRegex(RuntimeError, "Initialize or restore"):
            optimizer.refresh()
        self.assertFalse(optimizer.is_initialized)
        self.assertEqual(optimizer.num_refreshes, 0)

    @parameterized.expand([0, -1, True, False, 1.5, "2", float("nan")])
    def test_invalid_refresh_schedule_is_rejected(self, value):
        with self.assertRaisesRegex(ValueError, "refresh_every"):
            SST(torch.optim.SGD(self.model.parameters()), model=self.model, refresh_every=value)

    def test_refresh_checkpoint_preserves_the_clock_and_exact_training_trajectory(self):
        optimizer = SST(
            torch.optim.AdamW(self.model.parameters(), lr=0.05),
            model=self.model,
            rank_fraction=0.5,
            refresh_every=2,
        )
        gradients = [
            {name: torch.randn_like(parameter) for name, parameter in self.model.named_parameters()} for _ in range(6)
        ]
        for step, gradient in enumerate(gradients):
            for name, parameter in self.model.named_parameters():
                parameter.grad = gradient[name].clone()
            optimizer.step()
            if step == 2:
                checkpoint = copy.deepcopy(optimizer.state_dict())
                replacement_model = copy.deepcopy(self.model)
        with patch("torch.linalg.svd", side_effect=AssertionError("Loading must not recompute the active basis")):
            restored = SST(
                torch.optim.AdamW(replacement_model.parameters(), lr=1.0),
                model=replacement_model,
                initialize=False,
                refresh_every=1,
            )
            restored.load_state_dict(checkpoint)
        self.assertEqual((restored.refresh_every, restored.num_updates, restored.last_refresh_step), (2, 3, 2))
        for gradient in gradients[3:]:
            for name, parameter in replacement_model.named_parameters():
                parameter.grad = gradient[name].clone()
            restored.step()
        self.assert_state_equal(self.model.state_dict(), replacement_model.state_dict())
        self.assert_state_equal(optimizer.sst_state_dict(), restored.sst_state_dict())
        self.assert_state_equal(optimizer.optimizer.state_dict(), restored.optimizer.state_dict())
        self.assertEqual((restored.num_updates, restored.last_refresh_step, restored.num_refreshes), (6, 6, 3))

    def test_legacy_checkpoint_keeps_fixed_basis_semantics(self):
        optimizer = SST(torch.optim.SGD(self.model.parameters()), model=self.model, rank_fraction=0.5)
        legacy = copy.deepcopy(optimizer.sst_state_dict())
        legacy["version"] = 1
        for name in ("refresh_every", "num_updates", "last_refresh_step", "num_refreshes"):
            legacy.pop(name)
        restored = SST(torch.optim.SGD(self.model.parameters()), model=self.model, initialize=False, refresh_every=2)
        with patch("torch.linalg.svd", side_effect=AssertionError("Legacy restore must not run an SVD")):
            restored.load_sst_state_dict(legacy)
        self.assertIsNone(restored.refresh_every)
        self.assertEqual((restored.num_updates, restored.last_refresh_step, restored.num_refreshes), (0, 0, 0))
        self.assert_state_equal(legacy["protected_weights"], restored.sst_state_dict()["protected_weights"])

    def test_invalid_refresh_checkpoint_is_rejected_without_mutating_active_state(self):
        optimizer = SST(torch.optim.SGD(self.model.parameters()), model=self.model, rank_fraction=0.5)
        original = copy.deepcopy(optimizer.sst_state_dict())
        for field, value in (
            ("refresh_every", 0),
            ("refresh_every", True),
            ("num_updates", -1),
            ("num_updates", 1.5),
            ("num_refreshes", True),
            ("last_refresh_step", 1),
        ):
            with self.subTest(field=field, value=value):
                state = {**original, field: value}
                with self.assertRaisesRegex(ValueError, "refresh"):
                    optimizer.load_sst_state_dict(state)
                self.assert_state_equal(original, optimizer.sst_state_dict())
        for field in ("refresh_every", "num_updates", "last_refresh_step", "num_refreshes"):
            with self.subTest(missing=field):
                state = original.copy()
                state.pop(field)
                with self.assertRaisesRegex(ValueError, "refresh"):
                    optimizer.load_sst_state_dict(state)
                self.assert_state_equal(original, optimizer.sst_state_dict())

    def test_checkpoint_restores_basis_and_optimizer_without_svd(self):
        inner = torch.optim.AdamW(self.model.parameters(), lr=0.03)
        optimizer = SST(inner, model=self.model, rank_fraction=0.5)
        for parameter in self.model.parameters():
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        state = copy.deepcopy(optimizer.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer.pt"
            torch.save(state, path)
            state = torch.load(path, map_location="cpu", weights_only=True)
        replacement = torch.optim.AdamW(self.model.parameters(), lr=0.2)
        with patch("torch.linalg.svd", side_effect=AssertionError("Checkpoint restore must not run an SVD")):
            restored = SST(replacement, model=self.model, initialize=False)
            restored.load_state_dict(state)
        self.assertEqual(restored.rank_fraction, 0.5)
        self.assertIs(restored.state, replacement.state)
        self.assertIs(restored.param_groups, replacement.param_groups)
        self.assertEqual(replacement.param_groups[0]["lr"], 0.03)
        self.assert_state_equal(optimizer.sst_state_dict(), restored.sst_state_dict())
        for parameter in self.model.parameters():
            parameter.grad = torch.randn_like(parameter)
        restored.step()
        self.assert_protected(restored)

    def test_state_dictionary_hooks_and_extra_inner_state(self):
        class CustomOptimizer(torch.optim.SGD):
            def state_dict(self):
                return {**super().state_dict(), "custom_metadata": "preserved"}

            def load_state_dict(self, state_dict):
                self.loaded_metadata = state_dict["custom_metadata"]
                return super().load_state_dict(state_dict)

        inner = CustomOptimizer(self.model.parameters(), lr=0.1)
        optimizer = SST(inner, model=self.model, rank_fraction=0.5)
        calls = []
        optimizer.register_state_dict_pre_hook(lambda *_: calls.append("save_pre"))
        optimizer.register_state_dict_post_hook(lambda *_: calls.append("save_post"))
        optimizer.register_load_state_dict_pre_hook(lambda *_: calls.append("load_pre"))
        optimizer.register_load_state_dict_post_hook(lambda *_: calls.append("load_post"))
        state = optimizer.state_dict()
        optimizer.load_state_dict(state)
        self.assertEqual(calls, ["save_pre", "save_post", "load_pre", "load_post"])
        self.assertEqual(inner.loaded_metadata, "preserved")

    @parameterized.expand([(None,), (2,)])
    def test_deepcopy_preserves_wrapper_basis_and_refresh_clock(self, refresh_every):
        optimizer = SST(
            torch.optim.AdamW(self.model.parameters()),
            model=self.model,
            rank_fraction=0.5,
            refresh_every=refresh_every,
        )
        for parameter in self.model.parameters():
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        original = self.model.q_proj.weight.detach().clone()
        with patch("torch.linalg.svd", side_effect=AssertionError("Copy must not refresh the basis")):
            copied = copy.deepcopy(optimizer)
        self.assertIs(copied.param_groups, copied.optimizer.param_groups)
        self.assertIs(copied.state, copied.optimizer.state)
        self.assertIsNot(copied.model, self.model)
        self.assert_state_equal(optimizer.sst_state_dict(), copied.sst_state_dict())
        for parameter in copied.model.parameters():
            parameter.grad = torch.randn_like(parameter)
        copied.step()
        self.assertEqual(copied.num_updates, 2)
        self.assertEqual(copied.num_refreshes, 0 if refresh_every is None else 1)
        self.assertEqual(optimizer.num_updates, 1)
        self.assertEqual(optimizer.num_refreshes, 0)
        torch.testing.assert_close(self.model.q_proj.weight, original, rtol=0, atol=0)
        self.model = copied.model
        self.assert_protected(copied)

    def test_step_hooks(self):
        optimizer = SST(torch.optim.AdamW(self.model.parameters()), model=self.model, rank_fraction=0.5)
        calls = []
        optimizer.register_step_pre_hook(lambda *_: calls.append("before"))
        optimizer.register_step_post_hook(lambda *_: calls.append("after"))
        self.model.q_proj.weight.grad = torch.randn_like(self.model.q_proj.weight)
        optimizer.step()
        self.assertEqual(calls, ["before", "after"])

    def test_skipped_prepared_step_does_not_project(self):
        inner = torch.optim.SGD(self.model.parameters(), lr=0.1)
        inner.step_was_skipped = True
        optimizer = SST(inner, model=self.model, rank_fraction=0.5, refresh_every=1)
        with (
            patch.object(optimizer, "project", new=Mock()) as project,
            patch.object(optimizer, "refresh", new=Mock()) as refresh,
        ):
            optimizer.step()
        project.assert_not_called()
        refresh.assert_not_called()
        self.assertEqual(optimizer.num_updates, 0)

    def test_prepared_optimizer_accumulation_does_not_project(self):
        class AccumulatingOptimizer(torch.optim.SGD):
            gradient_state = SimpleNamespace(sync_gradients=False)

            def step(self, closure=None):
                if self.gradient_state.sync_gradients:
                    return super().step(closure)

        inner = AccumulatingOptimizer(self.model.parameters(), lr=0.1)
        optimizer = SST(inner, model=self.model, rank_fraction=0.5, refresh_every=1)
        self.model.q_proj.weight.grad = torch.randn_like(self.model.q_proj.weight)
        before = self.model.q_proj.weight.detach().clone()
        with patch.object(optimizer, "project", wraps=optimizer.project) as project:
            optimizer.step()
            project.assert_not_called()
            self.assertEqual((optimizer.num_updates, optimizer.num_refreshes), (0, 0))
            torch.testing.assert_close(self.model.q_proj.weight, before, rtol=0, atol=0)
            inner.gradient_state.sync_gradients = True
            optimizer.step()
            project.assert_called_once()
            self.assertEqual((optimizer.num_updates, optimizer.num_refreshes), (1, 1))
        self.assert_protected(optimizer)

    def test_differentiable_optimizer_is_rejected(self):
        for group_override in (True, False):
            groups = (
                [{"params": self.model.parameters(), "differentiable": True}]
                if group_override
                else self.model.parameters()
            )
            inner = torch.optim.SGD(groups, lr=0.1, differentiable=not group_override)
            with self.assertRaisesRegex(ValueError, "differentiable=True"):
                SST(inner, model=self.model)

    def test_engine_proxy_defaults_are_found_on_inner_optimizer(self):
        class PreparedOptimizer(torch.optim.SGD):
            def __init__(self, inner):
                super().__init__(inner.param_groups, **inner.defaults)
                self.param_groups = inner.param_groups
                self.state = inner.state
                del self.defaults
                self.optimizer = SimpleNamespace(optimizer=inner)

            def __getattr__(self, name):
                if name == "defaults":
                    return self.optimizer.defaults
                raise AttributeError(name)

            def step(self, closure=None):
                return self.optimizer.optimizer.step(closure)

        inner = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        prepared = PreparedOptimizer(inner)
        optimizer = SST(prepared, model=self.model, rank_fraction=0.5)
        self.assertIs(optimizer.defaults, inner.defaults)
        self.model.q_proj.weight.grad = torch.randn_like(self.model.q_proj.weight)
        optimizer.step()
        self.assert_protected(optimizer)

    def test_grad_scaler_overflow_does_not_step_or_project(self):
        inner = torch.optim.AdamW(self.model.parameters(), lr=0.1)
        optimizer = SST(inner, model=self.model, rank_fraction=0.5, refresh_every=1)
        scaler = torch.amp.GradScaler(torch.device(torch_device).type)
        loss = self.model.q_proj.weight.square().sum()
        scaler.scale(loss).backward()
        self.model.q_proj.weight.grad.fill_(float("inf"))
        before = self.model.q_proj.weight.detach().clone()
        with patch.object(optimizer, "project", new=Mock()) as project:
            scaler.step(optimizer)
            scaler.update()
        project.assert_not_called()
        torch.testing.assert_close(self.model.q_proj.weight, before, rtol=0, atol=0)
        self.assertEqual(len(inner.state), 0)
        self.assertEqual((optimizer.num_updates, optimizer.num_refreshes), (0, 0))

    @parameterized.expand(["float64", "complex64", "complex128"])
    def test_higher_precision_and_complex_parameters(self, dtype):
        self.model = _Model(dtype=getattr(torch, dtype)).to(torch_device)
        optimizer = SST(
            torch.optim.SGD(self.model.parameters(), lr=0.1),
            model=self.model,
            rank_fraction=0.5,
            refresh_every=1,
        )
        for parameter in self.model.parameters():
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        self.assert_protected(optimizer, tolerance=2e-12 if dtype in ("float64", "complex128") else 3e-6)

    @parameterized.expand([torch.float16, torch.bfloat16] if is_torch_available() else [])
    def test_low_precision_writeback_warns_once(self, dtype):
        self.model = _Model(dtype=dtype).to(torch_device)
        optimizer = SST(
            torch.optim.SGD(self.model.parameters(), lr=0.1),
            model=self.model,
            target_modules=["q_proj"],
            rank_fraction=0.5,
        )
        with self.assertWarnsRegex(UserWarning, "approximate"):
            self.model.q_proj.weight.grad = torch.randn_like(self.model.q_proj.weight)
            optimizer.step()
        with patch("warnings.warn") as warn:
            optimizer.step()
        warn.assert_not_called()

    @parameterized.expand([-1.0, 0.0, 1.0, 1.5, float("nan"), float("inf")])
    def test_invalid_rank_fraction(self, value):
        with self.assertRaisesRegex(ValueError, "rank_fraction"):
            SST(torch.optim.SGD(self.model.parameters(), lr=0.1), model=self.model, rank_fraction=value)

    def test_invalid_or_empty_targets(self):
        with self.assertRaisesRegex(ValueError, "No trainable"):
            SST(torch.optim.SGD(self.model.parameters(), lr=0.1), model=self.model, target_modules=["missing"])
        with self.assertRaisesRegex(ValueError, "model="):
            SST(torch.optim.SGD(self.model.parameters(), lr=0.1), target_modules=["q_proj"])
        with self.assertRaises(TypeError):
            SST(torch.optim.SGD(self.model.parameters(), lr=0.1), model=self.model, target_modules=[None])

    def test_deferred_initialization_cannot_silently_reanchor(self):
        optimizer = SST(torch.optim.SGD(self.model.parameters(), lr=0.1), model=self.model, initialize=False)
        with self.assertRaisesRegex(RuntimeError, "active basis"):
            optimizer.step()
        with self.assertRaisesRegex(ValueError, "active basis"):
            optimizer.load_state_dict(optimizer.optimizer.state_dict())
        optimizer.initialize()
        self.assertTrue(optimizer.is_initialized)

    def test_checkpoint_target_and_shape_validation(self):
        optimizer = SST(torch.optim.AdamW(self.model.parameters()), model=self.model, rank_fraction=0.5)
        state = optimizer.sst_state_dict()
        wrong_shape = copy.deepcopy(state)
        wrong_shape["protected_weights"][0]["left"] = torch.ones(9, 3)
        with self.assertRaisesRegex(ValueError, "shape"):
            optimizer.load_sst_state_dict(wrong_shape)
        missing = copy.deepcopy(state)
        missing["protected_weights"].pop()
        with self.assertRaisesRegex(ValueError, "targets"):
            optimizer.load_sst_state_dict(missing)
        self.assert_state_equal(state, optimizer.sst_state_dict())

    def test_checkpoint_rejects_invalid_basis_values(self):
        optimizer = SST(torch.optim.AdamW(self.model.parameters()), model=self.model, rank_fraction=0.5)
        original = copy.deepcopy(optimizer.sst_state_dict())
        for field, transform, error in (
            ("left", lambda tensor: tensor.to(torch.int64), "dtypes"),
            ("right", lambda tensor: tensor.double(), "dtypes"),
            ("singular_values", lambda tensor: -tensor, "Negative"),
            ("left", lambda tensor: torch.full_like(tensor, float("nan")), "Nonfinite"),
        ):
            state = copy.deepcopy(original)
            state["protected_weights"][0][field] = transform(state["protected_weights"][0][field])
            with self.assertRaisesRegex(ValueError, error):
                optimizer.load_sst_state_dict(state)
            self.assert_state_equal(original, optimizer.sst_state_dict())

    def test_native_implementation_does_not_import_external_sst(self):
        with patch.dict(
            "sys.modules", {"sst_torch": None, "sst_torch.projector": None, "sst_torch.transformers": None}
        ):
            optimizer = SST(torch.optim.SGD(self.model.parameters(), lr=0.1), model=self.model, rank_fraction=0.5)
            self.model.q_proj.weight.grad = torch.randn_like(self.model.q_proj.weight)
            optimizer.step()
            self.assert_protected(optimizer)
