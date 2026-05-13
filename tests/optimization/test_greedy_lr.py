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

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch
    from torch import nn

    from transformers.optimization import GreedyLR, StreamingAverage, get_greedy_schedule


@require_torch
class GreedyLRTest(unittest.TestCase):
    def _get_scheduler(self, **kwargs):
        model = nn.Linear(10, 10)
        defaults = {"lr": 0.1}
        defaults.update(kwargs.pop("optim_kwargs", {}))
        optimizer = torch.optim.SGD(model.parameters(), **defaults)
        scheduler_kwargs = {
            "mode": "min",
            "factor": 0.9,
            "patience": 3,
            "min_lr": 1e-6,
            "max_lr": 1.0,
            "cooldown": 0,
            "warmup": 0,
            "verbose": False,
        }
        scheduler_kwargs.update(kwargs)
        scheduler = GreedyLR(optimizer, **scheduler_kwargs)
        return optimizer, scheduler

    def test_initialization_valid_params(self):
        optimizer, scheduler = self._get_scheduler()
        self.assertEqual(scheduler.mode, "min")
        self.assertAlmostEqual(scheduler.factor, 0.9)
        self.assertEqual(scheduler.patience, 3)
        self.assertEqual(len(scheduler.min_lrs), len(optimizer.param_groups))
        self.assertEqual(len(scheduler.max_lrs), len(optimizer.param_groups))
        self.assertAlmostEqual(scheduler._last_lr[0], 0.1)

    def test_initialization_max_mode(self):
        optimizer, scheduler = self._get_scheduler(mode="max")
        self.assertEqual(scheduler.mode, "max")
        self.assertEqual(scheduler.best, float("-inf"))

    def test_initialization_invalid_factor(self):
        with self.assertRaises(ValueError):
            self._get_scheduler(factor=1.0)
        with self.assertRaises(ValueError):
            self._get_scheduler(factor=1.5)

    def test_initialization_invalid_mode(self):
        with self.assertRaises(ValueError):
            self._get_scheduler(mode="unknown")

    def test_initialization_invalid_threshold_mode(self):
        with self.assertRaises(ValueError):
            self._get_scheduler(threshold_mode="unknown")

    def test_initialization_not_optimizer(self):
        with self.assertRaises(TypeError):
            GreedyLR("not_an_optimizer")

    def test_lr_decrease_on_plateau(self):
        optimizer, scheduler = self._get_scheduler(patience=3)
        initial_lr = optimizer.param_groups[0]["lr"]

        # Establish a best metric
        scheduler.step(5.0)
        # Provide worse metrics for patience + 1 steps to trigger decrease
        for _ in range(4):
            scheduler.step(10.0)

        self.assertLess(optimizer.param_groups[0]["lr"], initial_lr)
        expected_lr = initial_lr * 0.9
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected_lr, places=7)

    def test_lr_increase_on_improvement(self):
        optimizer, scheduler = self._get_scheduler(patience=3)
        initial_lr = optimizer.param_groups[0]["lr"]

        # Provide continuously improving metrics for patience + 1 steps
        metric = 10.0
        for _ in range(4):
            metric *= 0.8
            scheduler.step(metric)

        self.assertGreater(optimizer.param_groups[0]["lr"], initial_lr)
        expected_lr = initial_lr / 0.9
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected_lr, places=7)

    def test_lr_never_below_min_lr(self):
        optimizer, scheduler = self._get_scheduler(patience=1, min_lr=0.01, factor=0.5)

        # Establish a best metric, then plateau repeatedly
        scheduler.step(1.0)
        for _ in range(50):
            scheduler.step(10.0)

        self.assertGreaterEqual(optimizer.param_groups[0]["lr"], 0.01 - 1e-10)

    def test_lr_never_above_max_lr(self):
        optimizer, scheduler = self._get_scheduler(patience=1, max_lr=0.2, factor=0.5, optim_kwargs={"lr": 0.1})

        # Provide continuously improving metrics
        metric = 10.0
        for _ in range(50):
            metric *= 0.8
            scheduler.step(metric)

        self.assertLessEqual(optimizer.param_groups[0]["lr"], 0.2 + 1e-10)

    def test_cooldown_prevents_further_reduction(self):
        optimizer, scheduler = self._get_scheduler(patience=2, cooldown=3)

        # Trigger a reduction
        scheduler.step(5.0)
        for _ in range(3):
            scheduler.step(10.0)

        lr_after_reduction = optimizer.param_groups[0]["lr"]

        # During cooldown, more bad metrics should NOT trigger another reduction
        for _ in range(3):
            scheduler.step(10.0)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], lr_after_reduction, places=7)

    def test_warmup_prevents_further_increase(self):
        optimizer, scheduler = self._get_scheduler(patience=2, warmup=3)

        # Trigger an increase
        metric = 10.0
        for _ in range(3):
            metric *= 0.8
            scheduler.step(metric)

        lr_after_increase = optimizer.param_groups[0]["lr"]

        # During warmup, more good metrics should NOT trigger another increase
        for _ in range(3):
            metric *= 0.8
            scheduler.step(metric)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], lr_after_increase, places=7)

    def test_smoothing_uses_streaming_average(self):
        optimizer, scheduler = self._get_scheduler(smooth=True, window_size=3, patience=10)

        self.assertIsNotNone(scheduler._streaming_avg)
        self.assertEqual(scheduler._streaming_avg.window_size, 3)

        scheduler.step(10.0)
        scheduler.step(8.0)
        scheduler.step(6.0)
        scheduler.step(4.0)

        # Window should be capped at size 3
        self.assertEqual(len(scheduler._streaming_avg.values), 3)
        # After 4 values with window 3, values are [8.0, 6.0, 4.0], avg = 6.0
        avg = scheduler._streaming_avg.sum / len(scheduler._streaming_avg.values)
        self.assertAlmostEqual(avg, 6.0, places=5)

    def test_no_smoothing_by_default(self):
        _, scheduler = self._get_scheduler()
        self.assertIsNone(scheduler._streaming_avg)

    def test_state_dict_round_trip(self):
        optimizer1, scheduler1 = self._get_scheduler(smooth=True, window_size=5)

        # Build up state
        metrics = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 6.0, 7.0]
        for m in metrics:
            scheduler1.step(m)

        state = scheduler1.state_dict()

        # Create a new scheduler and load state
        optimizer2, scheduler2 = self._get_scheduler(smooth=True, window_size=5)
        scheduler2.load_state_dict(state)

        self.assertEqual(scheduler2.best, scheduler1.best)
        self.assertEqual(scheduler2.num_bad_epochs, scheduler1.num_bad_epochs)
        self.assertEqual(scheduler2.num_good_epochs, scheduler1.num_good_epochs)
        self.assertEqual(scheduler2.last_epoch, scheduler1.last_epoch)
        self.assertEqual(scheduler2.cooldown_counter, scheduler1.cooldown_counter)
        self.assertEqual(scheduler2.warmup_counter, scheduler1.warmup_counter)
        self.assertAlmostEqual(optimizer2.param_groups[0]["lr"], optimizer1.param_groups[0]["lr"], places=7)

        # Both schedulers should behave identically going forward
        for m in [5.0, 4.0, 3.0]:
            scheduler1.step(m)
            scheduler2.step(m)
            self.assertAlmostEqual(optimizer1.param_groups[0]["lr"], optimizer2.param_groups[0]["lr"], places=7)

    def test_state_dict_contains_all_keys(self):
        _, scheduler = self._get_scheduler(smooth=True)
        scheduler.step(10.0)
        state = scheduler.state_dict()

        required_keys = [
            "factor",
            "min_lrs",
            "max_lrs",
            "patience",
            "verbose",
            "cooldown",
            "warmup",
            "cooldown_counter",
            "warmup_counter",
            "mode",
            "threshold",
            "threshold_mode",
            "best",
            "num_bad_epochs",
            "num_good_epochs",
            "eps",
            "last_epoch",
            "smooth",
            "window_size",
            "reset_start",
            "reset_start_original",
            "_last_lr",
            "_init_lrs",
            "_streaming_avg",
        ]
        for key in required_keys:
            self.assertIn(key, state)

    def test_load_state_dict_backward_compatibility(self):
        _, scheduler = self._get_scheduler()

        partial_state = {
            "factor": 0.8,
            "patience": 5,
            "best": 5.0,
            "num_bad_epochs": 3,
        }
        scheduler.load_state_dict(partial_state)

        self.assertAlmostEqual(scheduler.factor, 0.8)
        self.assertEqual(scheduler.patience, 5)
        self.assertAlmostEqual(scheduler.best, 5.0)
        self.assertEqual(scheduler.num_bad_epochs, 3)
        # Missing keys should retain defaults
        self.assertEqual(scheduler.cooldown_counter, 0)

    def test_factory_function(self):
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = get_greedy_schedule(optimizer, patience=5, min_lr=1e-5, factor=0.95)

        self.assertIsInstance(scheduler, GreedyLR)
        self.assertEqual(scheduler.patience, 5)
        self.assertAlmostEqual(scheduler.factor, 0.95)

    def test_factory_function_with_kwargs(self):
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = get_greedy_schedule(optimizer, mode="max", smooth=True, window_size=10)

        self.assertIsInstance(scheduler, GreedyLR)
        self.assertEqual(scheduler.mode, "max")
        self.assertTrue(scheduler.smooth)
        self.assertIsNotNone(scheduler._streaming_avg)

    def test_get_scheduler_integration(self):
        from transformers import get_scheduler

        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = get_scheduler(
            "greedy",
            optimizer=optimizer,
            scheduler_specific_kwargs={"patience": 5, "factor": 0.95},
        )
        self.assertIsInstance(scheduler, GreedyLR)
        self.assertEqual(scheduler.patience, 5)

    def test_get_last_lr(self):
        optimizer, scheduler = self._get_scheduler()
        scheduler.step(10.0)
        last_lr = scheduler.get_last_lr()
        self.assertIsInstance(last_lr, list)
        self.assertEqual(len(last_lr), len(optimizer.param_groups))

    def test_reset_at_min_lr(self):
        optimizer, scheduler = self._get_scheduler(patience=1, min_lr=0.01, factor=0.5, reset_start=2)
        initial_lr = optimizer.param_groups[0]["lr"]

        # Drive LR to min_lr
        scheduler.step(1.0)
        for _ in range(100):
            scheduler.step(10.0)

        # After reset, LR should be back to initial
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], initial_lr, places=7)

    def test_max_mode_lr_decrease(self):
        optimizer, scheduler = self._get_scheduler(mode="max", patience=2)
        initial_lr = optimizer.param_groups[0]["lr"]

        # Establish best, then provide worse (lower) metrics
        scheduler.step(10.0)
        for _ in range(3):
            scheduler.step(1.0)

        self.assertLess(optimizer.param_groups[0]["lr"], initial_lr)

    def test_max_mode_lr_increase(self):
        optimizer, scheduler = self._get_scheduler(mode="max", patience=2)
        initial_lr = optimizer.param_groups[0]["lr"]

        # Provide continuously improving (higher) metrics
        metric = 1.0
        for _ in range(3):
            metric *= 1.5
            scheduler.step(metric)

        self.assertGreater(optimizer.param_groups[0]["lr"], initial_lr)

    def test_relative_threshold_mode(self):
        optimizer, scheduler = self._get_scheduler(threshold_mode="rel", threshold=0.1, patience=2)

        # Best is 10.0. With rel threshold 0.1, improvement needs current < 10.0 * 0.9 = 9.0
        scheduler.step(10.0)
        # 9.5 is not good enough (9.5 > 9.0)
        scheduler.step(9.5)
        self.assertEqual(scheduler.num_bad_epochs, 1)

    def test_multiple_param_groups(self):
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD([{"params": model.weight, "lr": 0.1}, {"params": model.bias, "lr": 0.01}])
        scheduler = GreedyLR(optimizer, patience=2, factor=0.9, min_lr=1e-6)

        self.assertEqual(len(scheduler.min_lrs), 2)
        self.assertEqual(len(scheduler.max_lrs), 2)

        # Trigger reduction
        scheduler.step(5.0)
        for _ in range(3):
            scheduler.step(10.0)

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1 * 0.9, places=7)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 0.01 * 0.9, places=7)


@require_torch
class StreamingAverageTest(unittest.TestCase):
    def test_basic_average(self):
        avg = StreamingAverage(window_size=3)
        self.assertAlmostEqual(avg.streamavg(1.0), 1.0)
        self.assertAlmostEqual(avg.streamavg(2.0), 1.5)
        self.assertAlmostEqual(avg.streamavg(3.0), 2.0)
        # Window full, oldest drops
        self.assertAlmostEqual(avg.streamavg(4.0), 3.0)

    def test_state_dict_round_trip(self):
        avg1 = StreamingAverage(window_size=3)
        avg1.streamavg(1.0)
        avg1.streamavg(2.0)
        avg1.streamavg(3.0)

        state = avg1.state_dict()
        avg2 = StreamingAverage(window_size=3)
        avg2.load_state_dict(state)

        self.assertEqual(avg2.values, avg1.values)
        self.assertAlmostEqual(avg2.sum, avg1.sum)
        self.assertEqual(avg2.window_size, avg1.window_size)


@require_torch
class BackwardCompatibilityTest(unittest.TestCase):
    def test_default_lr_scheduler_type_unchanged(self):
        from transformers import TrainingArguments

        args = TrainingArguments(output_dir="./test_output")
        self.assertEqual(args.lr_scheduler_type, "linear")

    def test_existing_schedulers_still_work(self):
        from transformers import get_scheduler

        model = nn.Linear(10, 10)
        for sched_type in ["linear", "cosine", "constant", "constant_with_warmup"]:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = get_scheduler(
                name=sched_type,
                optimizer=optimizer,
                num_warmup_steps=5,
                num_training_steps=100,
            )
            self.assertIsNotNone(scheduler)
            # Run a few steps to verify it works
            for _ in range(5):
                optimizer.step()
                scheduler.step()
            self.assertGreaterEqual(optimizer.param_groups[0]["lr"], 0.0)


if __name__ == "__main__":
    unittest.main()
