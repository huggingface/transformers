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

"""Tests for the DataProducer protocol and its integration with Trainer."""

import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments
from transformers.data_producer import (
    AsyncDataProducer,
    BaseDataProducer,
    DataProducerCallback,
    ProducerConfig,
)
from transformers.trainer_callback import TrainerCallback


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class SimpleDataset(Dataset):
    """Minimal map-style dataset with synthetic (input_x, labels) data."""

    def __init__(self, length=64, seed=42):
        rng = np.random.RandomState(seed)
        self.x = rng.normal(size=(length,)).astype(np.float32)
        self.y = (2.0 * self.x + 3.0 + rng.normal(scale=0.1, size=(length,))).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"input_x": self.x[idx], "labels": self.y[idx]}


class RegressionModel(nn.Module):
    """Trivial y = ax + b model for testing."""

    def __init__(self, a=0.0, b=0.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))

    def forward(self, input_x, labels=None, **kwargs):
        y = input_x * self.a + self.b
        if labels is None:
            return (y,)
        loss = nn.functional.mse_loss(y, labels)
        return (loss, y)


class CountingProducer(BaseDataProducer):
    """Tracks produce() call counts and global steps."""

    def __init__(self, config=None, dataset_length=32):
        super().__init__(config)
        self.call_count = 0
        self.global_steps = []
        self.dataset_length = dataset_length

    def produce(self, model, global_step, **kwargs):
        self.call_count += 1
        self.global_steps.append(global_step)
        return SimpleDataset(length=self.dataset_length, seed=42 + self.call_count)


class LifecycleTrackingProducer(BaseDataProducer):
    """Tracks on_rollout_begin/end and produce calls."""

    def __init__(self, config=None):
        super().__init__(config)
        self.events = []

    def on_rollout_begin(self, global_step):
        self.events.append(("rollout_begin", global_step))

    def on_rollout_end(self, dataset, global_step):
        self.events.append(("rollout_end", global_step))

    def produce(self, model, global_step, **kwargs):
        self.events.append(("produce", global_step))
        return SimpleDataset(length=32)


def _make_trainer(data_producer, max_steps=10, **kwargs):
    """Helper to create a Trainer with a DataProducer."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = RegressionModel()
        args = TrainingArguments(
            output_dir=tmp_dir,
            max_steps=max_steps,
            per_device_train_batch_size=8,
            learning_rate=0.1,
            report_to="none",
            use_cpu=True,
            logging_steps=999,  # suppress logging noise
            save_strategy="no",
            **kwargs,
        )
        trainer = Trainer(
            model=model,
            args=args,
            data_producer=data_producer,
        )
        return trainer, tmp_dir


# ---------------------------------------------------------------------------
# Unit tests: ProducerConfig
# ---------------------------------------------------------------------------


class TestProducerConfig(unittest.TestCase):
    def test_defaults(self):
        config = ProducerConfig()
        self.assertEqual(config.mini_epochs, 1)
        self.assertIsNone(config.max_rollouts)
        self.assertIsNone(config.steps_per_generation)
        self.assertEqual(config.num_iterations, 1)
        self.assertFalse(config.async_prefetch)
        self.assertTrue(config.eval_during_produce)

    def test_custom_values(self):
        config = ProducerConfig(mini_epochs=3, max_rollouts=50, num_iterations=2)
        self.assertEqual(config.mini_epochs, 3)
        self.assertEqual(config.max_rollouts, 50)
        self.assertEqual(config.num_iterations, 2)

    def test_invalid_mini_epochs(self):
        with self.assertRaises(ValueError):
            ProducerConfig(mini_epochs=0)

    def test_invalid_max_rollouts(self):
        with self.assertRaises(ValueError):
            ProducerConfig(max_rollouts=0)

    def test_invalid_num_iterations(self):
        with self.assertRaises(ValueError):
            ProducerConfig(num_iterations=0)


# ---------------------------------------------------------------------------
# Unit tests: BaseDataProducer
# ---------------------------------------------------------------------------


class TestBaseDataProducer(unittest.TestCase):
    def test_default_config(self):
        class Dummy(BaseDataProducer):
            def produce(self, model, global_step, **kwargs):
                return SimpleDataset()

        p = Dummy()
        self.assertIsInstance(p.config, ProducerConfig)
        self.assertEqual(p.config.mini_epochs, 1)

    def test_custom_config(self):
        class Dummy(BaseDataProducer):
            def produce(self, model, global_step, **kwargs):
                return SimpleDataset()

        config = ProducerConfig(mini_epochs=3)
        p = Dummy(config)
        self.assertEqual(p.config.mini_epochs, 3)


# ---------------------------------------------------------------------------
# Unit tests: AsyncDataProducer
# ---------------------------------------------------------------------------


class TestAsyncDataProducer(unittest.TestCase):
    def test_wraps_inner(self):
        producer = CountingProducer()
        async_producer = AsyncDataProducer(producer)
        self.assertIs(async_producer.config, producer.config)

    def test_first_call_synchronous(self):
        producer = CountingProducer()
        async_producer = AsyncDataProducer(producer)
        model = RegressionModel()
        ds = async_producer.produce(model, global_step=0)
        self.assertIsInstance(ds, SimpleDataset)
        # First call: one sync produce + one prefetch = 2
        self.assertGreaterEqual(producer.call_count, 1)
        async_producer.shutdown()

    def test_lifecycle_forwarding(self):
        producer = LifecycleTrackingProducer()
        async_producer = AsyncDataProducer(producer)
        async_producer.on_rollout_begin(global_step=5)
        self.assertEqual(producer.events[-1], ("rollout_begin", 5))
        async_producer.shutdown()


# ---------------------------------------------------------------------------
# Unit tests: DataProducerCallback
# ---------------------------------------------------------------------------


class TestDataProducerCallback(unittest.TestCase):
    def test_is_trainer_callback(self):
        self.assertTrue(issubclass(DataProducerCallback, TrainerCallback))

    def test_instance_check(self):
        cb = DataProducerCallback()
        self.assertIsInstance(cb, TrainerCallback)


# ---------------------------------------------------------------------------
# Integration tests: Trainer with DataProducer
# ---------------------------------------------------------------------------


class TestTrainerWithDataProducer(unittest.TestCase):
    def test_invalid_data_producer_type(self):
        """data_producer without produce() method raises TypeError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(TypeError):
                Trainer(
                    model=RegressionModel(),
                    args=TrainingArguments(tmp_dir, max_steps=5, report_to="none", use_cpu=True),
                    data_producer="not a producer",
                )

    def test_both_dataset_and_producer_raises(self):
        """Cannot pass both train_dataset and data_producer."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                Trainer(
                    model=RegressionModel(),
                    args=TrainingArguments(tmp_dir, max_steps=5, report_to="none", use_cpu=True),
                    train_dataset=SimpleDataset(),
                    data_producer=CountingProducer(),
                )

    def test_requires_max_steps_or_max_rollouts(self):
        """data_producer without max_steps or max_rollouts raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                Trainer(
                    model=RegressionModel(),
                    args=TrainingArguments(tmp_dir, report_to="none", use_cpu=True),
                    data_producer=CountingProducer(),
                )

    def test_basic_online_training(self):
        """Basic online training with max_steps."""
        producer = CountingProducer(dataset_length=32)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = RegressionModel()
            args = TrainingArguments(
                output_dir=tmp_dir,
                max_steps=5,
                per_device_train_batch_size=8,
                learning_rate=0.1,
                report_to="none",
                use_cpu=True,
                save_strategy="no",
            )
            trainer = Trainer(model=model, args=args, data_producer=producer)
            trainer.train()
            self.assertEqual(trainer.state.global_step, 5)
            # produce() should have been called at least once
            self.assertGreaterEqual(producer.call_count, 1)

    def test_max_rollouts(self):
        """Training with max_rollouts stops after the specified number of rollouts."""
        config = ProducerConfig(max_rollouts=3, mini_epochs=1)
        producer = CountingProducer(config=config, dataset_length=16)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = RegressionModel()
            args = TrainingArguments(
                output_dir=tmp_dir,
                max_steps=999,  # large enough to not be the stopping condition
                per_device_train_batch_size=8,
                learning_rate=0.1,
                report_to="none",
                use_cpu=True,
                save_strategy="no",
            )
            trainer = Trainer(model=model, args=args, data_producer=producer)
            trainer.train()
            # produce() called once in compute_plan + (max_rollouts - 1) in iter_epochs
            self.assertEqual(producer.call_count, 3)

    def test_mini_epochs(self):
        """mini_epochs=2 yields 2 passes per rollout."""
        config = ProducerConfig(max_rollouts=2, mini_epochs=2)
        producer = CountingProducer(config=config, dataset_length=16)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = RegressionModel()
            args = TrainingArguments(
                output_dir=tmp_dir,
                max_steps=999,
                per_device_train_batch_size=8,
                learning_rate=0.1,
                report_to="none",
                use_cpu=True,
                save_strategy="no",
            )
            trainer = Trainer(model=model, args=args, data_producer=producer)
            trainer.train()
            # 2 rollouts × 1 produce each = 2 produce calls
            self.assertEqual(producer.call_count, 2)
            # But global_step should reflect 2 rollouts × 2 mini_epochs × steps_per_epoch
            # steps_per_epoch = 16 / 8 = 2
            # total = 2 * 2 * 2 = 8
            self.assertEqual(trainer.state.global_step, 8)

    def test_lifecycle_hooks(self):
        """on_rollout_begin and on_rollout_end are called around produce()."""
        config = ProducerConfig(max_rollouts=2)
        producer = LifecycleTrackingProducer(config=config)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = RegressionModel()
            args = TrainingArguments(
                output_dir=tmp_dir,
                max_steps=999,
                per_device_train_batch_size=8,
                learning_rate=0.1,
                report_to="none",
                use_cpu=True,
                save_strategy="no",
            )
            trainer = Trainer(model=model, args=args, data_producer=producer)
            trainer.train()
            # Should see rollout_begin, produce, rollout_end for each rollout
            event_types = [e[0] for e in producer.events]
            self.assertIn("rollout_begin", event_types)
            self.assertIn("produce", event_types)
            self.assertIn("rollout_end", event_types)

    def test_no_data_producer_uses_static_path(self):
        """Without data_producer, Trainer uses the static dataset path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = RegressionModel()
            args = TrainingArguments(
                output_dir=tmp_dir,
                max_steps=5,
                per_device_train_batch_size=8,
                learning_rate=0.1,
                report_to="none",
                use_cpu=True,
                save_strategy="no",
            )
            trainer = Trainer(
                model=model, args=args, train_dataset=SimpleDataset(),
            )
            trainer.train()
            self.assertEqual(trainer.state.global_step, 5)

    def test_loss_decreases(self):
        """Online training should decrease the loss."""
        producer = CountingProducer(dataset_length=64)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = RegressionModel()
            args = TrainingArguments(
                output_dir=tmp_dir,
                max_steps=20,
                per_device_train_batch_size=16,
                learning_rate=0.5,
                report_to="none",
                use_cpu=True,
                save_strategy="no",
                logging_steps=5,
            )
            trainer = Trainer(model=model, args=args, data_producer=producer)
            trainer.train()
            # Check that loss decreased
            logs = trainer.state.log_history
            losses = [log["loss"] for log in logs if "loss" in log]
            self.assertGreater(len(losses), 1)
            self.assertLess(losses[-1], losses[0])

    def test_produce_receives_kwargs(self):
        """produce() receives processing_class, accelerator, args."""

        class InspectingProducer(BaseDataProducer):
            def __init__(self):
                super().__init__()
                self.received_kwargs = {}

            def produce(self, model, global_step, **kwargs):
                self.received_kwargs = kwargs
                return SimpleDataset(length=16)

        producer = InspectingProducer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = RegressionModel()
            args = TrainingArguments(
                output_dir=tmp_dir, max_steps=2, per_device_train_batch_size=8,
                report_to="none", use_cpu=True, save_strategy="no",
            )
            trainer = Trainer(model=model, args=args, data_producer=producer)
            trainer.train()
            self.assertIn("processing_class", producer.received_kwargs)
            self.assertIn("accelerator", producer.received_kwargs)
            self.assertIn("args", producer.received_kwargs)

    def test_callback_producer_registered(self):
        """A producer that inherits DataProducerCallback is registered as a Trainer callback."""

        class CallbackProducer(BaseDataProducer, DataProducerCallback):
            def produce(self, model, global_step, **kwargs):
                return SimpleDataset(length=16)

        producer = CallbackProducer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = RegressionModel()
            args = TrainingArguments(
                output_dir=tmp_dir, max_steps=2, per_device_train_batch_size=8,
                report_to="none", use_cpu=True, save_strategy="no",
            )
            trainer = Trainer(model=model, args=args, data_producer=producer)
            # The producer should be in the callback list
            callback_types = [type(cb) for cb in trainer.callback_handler.callbacks]
            self.assertIn(CallbackProducer, callback_types)


if __name__ == "__main__":
    unittest.main()
