# Copyright 2020-present the HuggingFace Inc. team.
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
"""Tests for the DataProducer protocol and online training support."""

import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset

from transformers import Trainer, TrainingArguments
from transformers.data_producer import (
    AsyncDataProducer,
    BaseDataProducer,
    DataProducer,
    DataProducerCallback,
    PreferencePairDataset,
    ProducerConfig,
    RolloutDataset,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class SimpleDataset(Dataset):
    """A minimal dataset that returns (input_x, labels) pairs."""

    def __init__(self, length=16, seed=42):
        rng = np.random.RandomState(seed)
        self.x = rng.normal(size=(length,)).astype(np.float32)
        self.y = (2.0 * self.x + 3.0 + rng.normal(scale=0.1, size=(length,))).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"input_x": self.x[idx], "labels": self.y[idx]}


class SimpleIterableDataset(IterableDataset):
    """An IterableDataset that yields a fixed number of items."""

    def __init__(self, length=16, seed=42):
        self.length = length
        self.seed = seed

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        for _ in range(self.length):
            x = np.float32(rng.normal())
            y = np.float32(2.0 * x + 3.0)
            yield {"input_x": x, "labels": y}


class RegressionModel(nn.Module):
    """A trivial y = ax + b model for testing."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.config = None

    def forward(self, input_x, labels=None, **kwargs):
        y = input_x * self.a + self.b
        if labels is None:
            return (y,)
        loss = nn.functional.mse_loss(y, labels)
        return (loss, y)


class CountingProducer(BaseDataProducer):
    """A DataProducer that counts how many times produce() is called."""

    def __init__(self, config=None, dataset_size=16):
        super().__init__(config or ProducerConfig())
        self.call_count = 0
        self.global_steps_seen = []
        self.dataset_size = dataset_size

    def produce(self, model, global_step, **kwargs):
        self.call_count += 1
        self.global_steps_seen.append(global_step)
        return SimpleDataset(length=self.dataset_size)


class IterableProducer(BaseDataProducer):
    """A DataProducer that returns an IterableDataset."""

    def __init__(self, config=None, dataset_size=16):
        super().__init__(config or ProducerConfig())
        self.dataset_size = dataset_size

    def produce(self, model, global_step, **kwargs):
        return SimpleIterableDataset(length=self.dataset_size)


class LifecycleTrackingProducer(BaseDataProducer):
    """Tracks lifecycle hook calls."""

    def __init__(self, config=None):
        super().__init__(config or ProducerConfig(max_rollouts=2))
        self.rollout_begins = []
        self.rollout_ends = []
        self.produce_calls = []

    def on_rollout_begin(self, global_step):
        self.rollout_begins.append(global_step)

    def on_rollout_end(self, dataset, global_step):
        self.rollout_ends.append(global_step)

    def produce(self, model, global_step, **kwargs):
        self.produce_calls.append(global_step)
        return SimpleDataset(length=8)


def _make_trainer(model=None, data_producer=None, max_steps=10, **kwargs):
    """Helper to create a Trainer with a DataProducer."""
    if model is None:
        model = RegressionModel()
    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(
            output_dir=tmp,
            max_steps=max_steps,
            per_device_train_batch_size=4,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            use_cpu=True,
            **kwargs,
        )
        trainer = Trainer(
            model=model,
            args=args,
            data_producer=data_producer,
        )
        yield trainer


# ---------------------------------------------------------------------------
# Unit tests: data_producer.py classes
# ---------------------------------------------------------------------------


class TestProducerConfig(unittest.TestCase):
    def test_defaults(self):
        config = ProducerConfig()
        self.assertEqual(config.mini_epochs, 1)
        self.assertIsNone(config.max_rollouts)
        self.assertFalse(config.async_prefetch)
        self.assertTrue(config.eval_during_produce)
        self.assertFalse(config.empty_cache_before_produce)
        self.assertFalse(config.empty_cache_after_produce)

    def test_custom_values(self):
        config = ProducerConfig(mini_epochs=3, max_rollouts=50, async_prefetch=True)
        self.assertEqual(config.mini_epochs, 3)
        self.assertEqual(config.max_rollouts, 50)
        self.assertTrue(config.async_prefetch)


class TestRolloutDataset(unittest.TestCase):
    def test_basic(self):
        prompts = [torch.tensor([1, 2, 3])] * 4
        completions = [torch.tensor([4, 5, 6])] * 4
        rewards = [1.0, 0.5, 0.8, 0.3]
        ds = RolloutDataset(prompts=prompts, completions=completions, rewards=rewards)
        self.assertEqual(len(ds), 4)
        item = ds[0]
        self.assertIn("prompt", item)
        self.assertIn("completion", item)
        self.assertIn("reward", item)

    def test_with_extras(self):
        prompts = [torch.tensor([1])] * 3
        completions = [torch.tensor([2])] * 3
        rewards = [1.0, 0.5, 0.8]
        extras = {"advantages": [0.1, 0.2, 0.3]}
        ds = RolloutDataset(prompts=prompts, completions=completions, rewards=rewards, extras=extras)
        item = ds[1]
        self.assertAlmostEqual(item["advantages"], 0.2)

    def test_length_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            RolloutDataset(prompts=[1, 2], completions=[1], rewards=[1, 2])


class TestPreferencePairDataset(unittest.TestCase):
    def test_basic(self):
        prompts = [torch.tensor([1])] * 3
        chosen = [torch.tensor([2])] * 3
        rejected = [torch.tensor([3])] * 3
        ds = PreferencePairDataset(prompts=prompts, chosen=chosen, rejected=rejected)
        self.assertEqual(len(ds), 3)
        item = ds[0]
        self.assertIn("prompt", item)
        self.assertIn("chosen", item)
        self.assertIn("rejected", item)

    def test_length_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            PreferencePairDataset(prompts=[1, 2], chosen=[1], rejected=[1, 2])


class TestBaseDataProducer(unittest.TestCase):
    def test_default_config(self):
        producer = CountingProducer()
        self.assertIsNotNone(producer.config)
        self.assertEqual(producer.config.mini_epochs, 1)

    def test_custom_config(self):
        config = ProducerConfig(mini_epochs=3)
        producer = CountingProducer(config=config)
        self.assertEqual(producer.config.mini_epochs, 3)

    def test_lifecycle_hooks_are_noop(self):
        producer = CountingProducer()
        # Should not raise
        producer.on_rollout_begin(global_step=0)
        producer.on_rollout_end(dataset=SimpleDataset(), global_step=0)


class TestAsyncDataProducer(unittest.TestCase):
    def test_wraps_inner(self):
        inner = CountingProducer(config=ProducerConfig(max_rollouts=5))
        async_producer = AsyncDataProducer(inner)
        self.assertEqual(async_producer.config.max_rollouts, 5)

    def test_first_call_synchronous(self):
        inner = CountingProducer(config=ProducerConfig(max_rollouts=5))
        async_producer = AsyncDataProducer(inner)
        model = RegressionModel()
        dataset = async_producer.produce(model=model, global_step=0)
        self.assertIsNotNone(dataset)
        self.assertEqual(inner.call_count, 2)  # 1 sync + 1 prefetch started

    def test_forwards_lifecycle_hooks(self):
        inner = LifecycleTrackingProducer()
        async_producer = AsyncDataProducer(inner)
        async_producer.on_rollout_begin(global_step=5)
        self.assertEqual(inner.rollout_begins, [5])


class TestDataProducerCallback(unittest.TestCase):
    def test_is_trainer_callback(self):
        from transformers.trainer_callback import TrainerCallback

        producer = CountingProducer()
        callback = DataProducerCallback(producer)
        self.assertIsInstance(callback, TrainerCallback)


# ---------------------------------------------------------------------------
# Integration tests: Trainer + DataProducer
# ---------------------------------------------------------------------------


class TestTrainerWithDataProducer(unittest.TestCase):
    def test_invalid_data_producer_type(self):
        """Passing a non-DataProducer should raise TypeError."""
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp, max_steps=1, report_to="none", use_cpu=True
            )
            with self.assertRaises(TypeError):
                Trainer(model=model, args=args, data_producer="not a producer")

    def test_basic_online_training(self):
        """DataProducer with max_rollouts=3 should train successfully."""
        producer = CountingProducer(
            config=ProducerConfig(max_rollouts=3),
            dataset_size=8,
        )
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                max_steps=6,
                per_device_train_batch_size=4,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            trainer = Trainer(model=model, args=args, data_producer=producer)
            result = trainer.train()

            # 3 rollouts: 1 in compute_plan + 2 in iter_epochs
            self.assertEqual(producer.call_count, 3)
            self.assertEqual(result.global_step, 6)

    def test_mini_epochs(self):
        """mini_epochs=2 should yield 2 training passes per rollout."""
        producer = CountingProducer(
            config=ProducerConfig(mini_epochs=2, max_rollouts=2),
            dataset_size=8,
        )
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                max_steps=8,  # 2 steps/epoch × 2 mini_epochs × 2 rollouts = 8
                per_device_train_batch_size=4,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            trainer = Trainer(model=model, args=args, data_producer=producer)
            result = trainer.train()

            # 2 rollouts: 1 in compute_plan + 1 in iter_epochs
            self.assertEqual(producer.call_count, 2)
            self.assertEqual(result.global_step, 8)

    def test_max_steps_stops_training(self):
        """Training should stop at max_steps even if max_rollouts allows more."""
        producer = CountingProducer(
            config=ProducerConfig(max_rollouts=100),
            dataset_size=8,
        )
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                max_steps=4,
                per_device_train_batch_size=4,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            trainer = Trainer(model=model, args=args, data_producer=producer)
            result = trainer.train()
            self.assertEqual(result.global_step, 4)

    def test_lifecycle_hooks_called(self):
        """on_rollout_begin and on_rollout_end should be called for each produce()."""
        producer = LifecycleTrackingProducer(
            config=ProducerConfig(max_rollouts=2),
        )
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                max_steps=4,
                per_device_train_batch_size=4,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            trainer = Trainer(model=model, args=args, data_producer=producer)
            trainer.train()

            # 2 produce calls: 1 in compute_plan + 1 in iter_epochs
            self.assertEqual(len(producer.rollout_begins), 2)
            self.assertEqual(len(producer.rollout_ends), 2)
            self.assertEqual(len(producer.produce_calls), 2)

    def test_eval_during_produce(self):
        """Model should be in eval mode during produce() if config says so."""
        model_modes = []

        class ModeTrackingProducer(BaseDataProducer):
            def produce(self, model, global_step, **kwargs):
                model_modes.append(model.training)
                return SimpleDataset(length=8)

        producer = ModeTrackingProducer(
            config=ProducerConfig(max_rollouts=2, eval_during_produce=True)
        )
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                max_steps=4,
                per_device_train_batch_size=4,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            trainer = Trainer(model=model, args=args, data_producer=producer)
            trainer.train()

            # Model should have been in eval mode during produce
            for mode in model_modes:
                self.assertFalse(mode, "Model should be in eval mode during produce()")

    def test_no_eval_during_produce(self):
        """Model should stay in training mode if eval_during_produce=False."""
        model_modes = []

        class ModeTrackingProducer(BaseDataProducer):
            def produce(self, model, global_step, **kwargs):
                model_modes.append(model.training)
                return SimpleDataset(length=8)

        producer = ModeTrackingProducer(
            config=ProducerConfig(max_rollouts=2, eval_during_produce=False)
        )
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                max_steps=4,
                per_device_train_batch_size=4,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            trainer = Trainer(model=model, args=args, data_producer=producer)
            trainer.train()

            # All calls after the first (compute_plan, model not yet in train mode)
            # should have model in training mode
            # The first produce is in compute_plan before model.train(), so skip it
            for mode in model_modes[1:]:
                self.assertTrue(mode, "Model should be in training mode during produce()")

    def test_async_prefetch_wrapping(self):
        """Setting async_prefetch=True should wrap the producer."""
        producer = CountingProducer(
            config=ProducerConfig(max_rollouts=2, async_prefetch=True),
            dataset_size=8,
        )
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                max_steps=4,
                per_device_train_batch_size=4,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            trainer = Trainer(model=model, args=args, data_producer=producer)
            self.assertIsInstance(trainer.data_producer, AsyncDataProducer)

    def test_no_data_producer_uses_static_path(self):
        """Without data_producer, the static training path should work."""
        model = RegressionModel()
        ds = SimpleDataset(length=16)
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                num_train_epochs=2,
                per_device_train_batch_size=4,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            trainer = Trainer(model=model, args=args, train_dataset=ds)
            result = trainer.train()
            self.assertGreater(result.global_step, 0)

    def test_requires_max_steps_or_max_rollouts(self):
        """Without max_steps or max_rollouts, should raise ValueError."""
        producer = CountingProducer(
            config=ProducerConfig(max_rollouts=None),
            dataset_size=8,
        )
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                max_steps=-1,
                per_device_train_batch_size=4,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            trainer = Trainer(model=model, args=args, data_producer=producer)
            with self.assertRaises(ValueError):
                trainer.train()

    def test_iterable_dataset_warning(self):
        """IterableDataset with mini_epochs > 1 should log a warning."""
        import logging

        producer = IterableProducer(
            config=ProducerConfig(mini_epochs=2, max_rollouts=1),
            dataset_size=8,
        )
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                max_steps=4,
                per_device_train_batch_size=4,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            trainer = Trainer(model=model, args=args, data_producer=producer)
            with self.assertLogs("transformers.trainer", level=logging.WARNING) as cm:
                trainer.train()
            warning_found = any("IterableDataset" in msg and "mini_epochs" in msg for msg in cm.output)
            self.assertTrue(warning_found, "Expected warning about IterableDataset + mini_epochs")

    def test_produce_receives_kwargs(self):
        """produce() should receive processing_class and accelerator."""
        received_kwargs = {}

        class KwargsTrackingProducer(BaseDataProducer):
            def produce(self, model, global_step, processing_class=None, accelerator=None, args=None, **kwargs):
                received_kwargs["processing_class"] = processing_class
                received_kwargs["accelerator"] = accelerator
                received_kwargs["args"] = args
                return SimpleDataset(length=8)

        producer = KwargsTrackingProducer(config=ProducerConfig(max_rollouts=1))
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                max_steps=2,
                per_device_train_batch_size=4,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            trainer = Trainer(model=model, args=args, data_producer=producer, processing_class="test_tokenizer")
            trainer.train()

            self.assertEqual(received_kwargs["processing_class"], "test_tokenizer")
            self.assertIsNotNone(received_kwargs["accelerator"])
            self.assertIsNotNone(received_kwargs["args"])

    def test_loss_decreases_with_online_training(self):
        """Online training should produce decreasing loss over steps."""
        producer = CountingProducer(
            config=ProducerConfig(max_rollouts=5),
            dataset_size=16,
        )
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                max_steps=20,
                per_device_train_batch_size=4,
                learning_rate=0.1,
                logging_steps=5,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            trainer = Trainer(model=model, args=args, data_producer=producer)
            result = trainer.train()
            self.assertEqual(result.global_step, 20)
            # Loss should be finite
            self.assertTrue(np.isfinite(result.training_loss))


# ---------------------------------------------------------------------------
# Integration tests: eval_data_producer & test_data_producer
# ---------------------------------------------------------------------------


class TestTrainerWithEvalDataProducer(unittest.TestCase):
    def test_eval_data_producer_basic(self):
        """eval_data_producer.produce() should be called during evaluate()."""
        eval_producer = CountingProducer(config=ProducerConfig(), dataset_size=8)
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                max_steps=1,
                per_device_train_batch_size=4,
                eval_strategy="no",
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            train_ds = SimpleDataset(length=8)
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_data_producer=eval_producer,
            )
            metrics = trainer.evaluate()
            self.assertEqual(eval_producer.call_count, 1)
            self.assertIn("eval_loss", metrics)

    def test_eval_data_producer_during_training(self):
        """eval_data_producer should be called at eval steps during training."""
        eval_producer = CountingProducer(config=ProducerConfig(), dataset_size=8)
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                max_steps=4,
                per_device_train_batch_size=4,
                eval_strategy="steps",
                eval_steps=2,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            train_ds = SimpleDataset(length=16)
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_data_producer=eval_producer,
            )
            trainer.train()
            # eval at step 2 and step 4 = 2 calls
            self.assertEqual(eval_producer.call_count, 2)

    def test_explicit_eval_dataset_overrides_producer(self):
        """Passing eval_dataset to evaluate() should override eval_data_producer."""
        eval_producer = CountingProducer(config=ProducerConfig(), dataset_size=8)
        model = RegressionModel()
        explicit_ds = SimpleDataset(length=8)
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                max_steps=1,
                per_device_train_batch_size=4,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            train_ds = SimpleDataset(length=8)
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_data_producer=eval_producer,
            )
            metrics = trainer.evaluate(eval_dataset=explicit_ds)
            # Producer should NOT have been called since explicit dataset was provided
            self.assertEqual(eval_producer.call_count, 0)
            self.assertIn("eval_loss", metrics)

    def test_static_eval_dataset_takes_priority_over_producer(self):
        """self.eval_dataset should take priority over eval_data_producer."""
        eval_producer = CountingProducer(config=ProducerConfig(), dataset_size=8)
        model = RegressionModel()
        static_eval_ds = SimpleDataset(length=8)
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                max_steps=1,
                per_device_train_batch_size=4,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            train_ds = SimpleDataset(length=8)
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=static_eval_ds,
                eval_data_producer=eval_producer,
            )
            metrics = trainer.evaluate()
            # Producer should NOT have been called since self.eval_dataset exists
            self.assertEqual(eval_producer.call_count, 0)
            self.assertIn("eval_loss", metrics)

    def test_invalid_eval_data_producer_type(self):
        """Passing a non-DataProducer as eval_data_producer should raise TypeError."""
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp, max_steps=1, report_to="none", use_cpu=True
            )
            with self.assertRaises(TypeError):
                Trainer(model=model, args=args, eval_data_producer="not a producer")

    def test_eval_strategy_accepts_eval_data_producer(self):
        """eval_strategy should not raise when eval_data_producer is set but eval_dataset is None."""
        eval_producer = CountingProducer(config=ProducerConfig(), dataset_size=8)
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                max_steps=2,
                per_device_train_batch_size=4,
                eval_strategy="steps",
                eval_steps=1,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            train_ds = SimpleDataset(length=8)
            # Should NOT raise ValueError about missing eval_dataset
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_data_producer=eval_producer,
            )
            trainer.train()
            self.assertGreater(eval_producer.call_count, 0)


class TestTrainerWithTestDataProducer(unittest.TestCase):
    def test_test_data_producer_basic(self):
        """test_data_producer.produce() should be called during predict()."""
        test_producer = CountingProducer(config=ProducerConfig(), dataset_size=8)
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                max_steps=1,
                per_device_train_batch_size=4,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            train_ds = SimpleDataset(length=8)
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                test_data_producer=test_producer,
            )
            output = trainer.predict()
            self.assertEqual(test_producer.call_count, 1)
            self.assertIsNotNone(output.predictions)

    def test_explicit_test_dataset_overrides_producer(self):
        """Passing test_dataset to predict() should override test_data_producer."""
        test_producer = CountingProducer(config=ProducerConfig(), dataset_size=8)
        model = RegressionModel()
        explicit_ds = SimpleDataset(length=8)
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                max_steps=1,
                per_device_train_batch_size=4,
                save_strategy="no",
                report_to="none",
                use_cpu=True,
            )
            train_ds = SimpleDataset(length=8)
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                test_data_producer=test_producer,
            )
            output = trainer.predict(test_dataset=explicit_ds)
            # Producer should NOT have been called
            self.assertEqual(test_producer.call_count, 0)
            self.assertIsNotNone(output.predictions)

    def test_predict_raises_without_dataset_or_producer(self):
        """predict() with no test_dataset and no test_data_producer should raise."""
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp, max_steps=1, report_to="none", use_cpu=True
            )
            train_ds = SimpleDataset(length=8)
            trainer = Trainer(model=model, args=args, train_dataset=train_ds)
            with self.assertRaises(ValueError):
                trainer.predict()

    def test_invalid_test_data_producer_type(self):
        """Passing a non-DataProducer as test_data_producer should raise TypeError."""
        model = RegressionModel()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp, max_steps=1, report_to="none", use_cpu=True
            )
            with self.assertRaises(TypeError):
                Trainer(model=model, args=args, test_data_producer=42)


if __name__ == "__main__":
    unittest.main()
