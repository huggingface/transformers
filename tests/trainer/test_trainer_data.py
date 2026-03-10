# Copyright 2018 the HuggingFace Inc. team.
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

"""
Trainer data-related tests: dataloaders, samplers, sharding, label smoothing,
batch size finder, pad/concatenate, collators, and eval loop container.
"""

import copy
import tempfile
import unittest
import warnings

import numpy as np
import torch
from torch import nn

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import default_data_collator as _default_data_collator
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.testing_utils import (
    TestCasePlus,
    backend_device_count,
    require_accelerate,
    require_torch,
    torch_device,
)
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    EvalLoopContainer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    ShardSampler,
    get_parameter_names,
    numpy_pad_and_concatenate,
    torch_pad_and_concatenate,
)
from transformers.trainer_utils import RemoveColumnsCollator, find_executable_batch_size

from .trainer_test_utils import (
    AlmostAccuracy,
    CustomDataloaderTrainer,
    DynamicShapesDataset,
    RegressionDataset,
    RegressionModel,
    RegressionModelConfig,
    RegressionPreTrainedModel,
    RegressionTrainingArguments,
    SampleIterableDataset,
    TrainerIntegrationCommon,
    TstLayer,
    get_regression_trainer,
)


class RandomIterableDataset(torch.utils.data.IterableDataset):
    # For testing, an iterable dataset of random length
    def __init__(self, p_stop=0.01, max_length=1000):
        self.p_stop = p_stop
        self.max_length = max_length
        self.generator = torch.Generator()

    def __iter__(self):
        count = 0
        stop = False
        while not stop and count < self.max_length:
            yield count
            count += 1
            number = torch.rand(1, generator=self.generator).item()
            stop = number < self.p_stop


# ---------------------------------------------------------------------------
# Dataloader tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerDataloaderTest(TestCasePlus):
    """Tests for train/eval dataloaders, drop_last, persistent workers."""

    def test_train_and_eval_dataloaders(self):
        if torch_device == "cuda":
            n_gpu = max(1, backend_device_count(torch_device))
        else:
            # DP is deprecated by PyTorch, accelerators like XPU doesn't support DP
            n_gpu = 1

        tmp_dir = self.get_auto_remove_tmp_dir()
        trainer = get_regression_trainer(learning_rate=0.1, per_device_train_batch_size=16, output_dir=tmp_dir)
        self.assertEqual(trainer.get_train_dataloader().total_batch_size, 16 * n_gpu)
        trainer = get_regression_trainer(learning_rate=0.1, per_device_eval_batch_size=16, output_dir=tmp_dir)
        self.assertEqual(trainer.get_eval_dataloader().total_batch_size, 16 * n_gpu)

        # Check drop_last works
        trainer = get_regression_trainer(
            train_len=66,
            eval_len=74,
            learning_rate=0.1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            output_dir=tmp_dir,
        )
        self.assertEqual(len(trainer.get_train_dataloader()), 66 // (16 * n_gpu) + 1)
        self.assertEqual(len(trainer.get_eval_dataloader()), 74 // (32 * n_gpu) + 1)

        trainer = get_regression_trainer(
            train_len=66,
            eval_len=74,
            learning_rate=0.1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            dataloader_drop_last=True,
            output_dir=tmp_dir,
        )
        self.assertEqual(len(trainer.get_train_dataloader()), 66 // (16 * n_gpu))
        self.assertEqual(len(trainer.get_eval_dataloader()), 74 // (32 * n_gpu))

        # Check passing a new dataset for evaluation works
        new_eval_dataset = RegressionDataset(length=128)
        self.assertEqual(len(trainer.get_eval_dataloader(new_eval_dataset)), 128 // (32 * n_gpu))

    # tests that we do not require dataloader to have a .dataset attribute
    def test_dataloader_without_dataset(self):
        train_dataset = RegressionDataset(length=128)
        trainer = CustomDataloaderTrainer(
            model=RegressionModel(),
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            args=TrainingArguments(output_dir=self.get_auto_remove_tmp_dir()),
        )

        trainer.train()
        trainer.evaluate()

    def test_get_eval_dataloader_without_persistent_workers(self):
        train_dataset = RegressionDataset()
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
        tiny_gpt2 = GPT2LMHeadModel(config)
        args = TrainingArguments(self.get_auto_remove_tmp_dir(), dataloader_persistent_workers=False)

        # Single evaluation dataset
        eval_dataset = RegressionDataset()
        trainer = Trainer(tiny_gpt2, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
        # Mocking the prepare method to avoid the dataloader changing with each call to get_eval_dataloader
        trainer.accelerator.prepare = lambda x: x

        default_dataloader = trainer.get_eval_dataloader()
        dataloader_with_dataset = trainer.get_eval_dataloader(eval_dataset)

        self.assertEqual(default_dataloader.dataset, eval_dataset)
        self.assertEqual(dataloader_with_dataset.dataset, eval_dataset)
        self.assertNotEqual(default_dataloader, dataloader_with_dataset)

        # Multiple evaluation datasets
        first_dataset = RegressionDataset()
        second_dataset = RegressionDataset()
        trainer = Trainer(
            tiny_gpt2,
            args,
            train_dataset=train_dataset,
            eval_dataset={"first": first_dataset, "second": second_dataset},
        )
        # Mocking the prepare method to avoid the dataloader changing with each call to get_eval_dataloader
        trainer.accelerator.prepare = lambda x: x

        first_dataloader = trainer.get_eval_dataloader("first")
        first_dataloader_repeated = trainer.get_eval_dataloader("first")
        second_dataloader = trainer.get_eval_dataloader("second")
        second_dataloader_repeated = trainer.get_eval_dataloader("second")

        self.assertEqual(first_dataset, first_dataloader.dataset)
        self.assertEqual(first_dataloader.dataset, first_dataloader_repeated.dataset)
        self.assertEqual(second_dataset, second_dataloader.dataset)
        self.assertEqual(second_dataloader.dataset, second_dataloader_repeated.dataset)
        self.assertNotEqual(first_dataloader, first_dataloader_repeated)
        self.assertNotEqual(second_dataloader, second_dataloader_repeated)

    def test_get_eval_dataloader_with_persistent_workers(self):
        train_dataset = RegressionDataset()
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
        tiny_gpt2 = GPT2LMHeadModel(config)
        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            dataloader_persistent_workers=True,
            dataloader_num_workers=2,
        )

        # Single evaluation dataset
        eval_dataset = RegressionDataset()
        trainer = Trainer(tiny_gpt2, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
        # Mocking the prepare method to avoid the dataloader changing with each call to get_eval_dataloader
        trainer.accelerator.prepare = lambda x: x

        default_dataloader = trainer.get_eval_dataloader()
        dataloader_with_dataset = trainer.get_eval_dataloader(eval_dataset)

        self.assertEqual(default_dataloader.dataset, eval_dataset)
        self.assertEqual(dataloader_with_dataset.dataset, eval_dataset)
        self.assertEqual(default_dataloader, dataloader_with_dataset)

        # Multiple evaluation datasets
        first_dataset = RegressionDataset()
        second_dataset = RegressionDataset()
        trainer = Trainer(
            tiny_gpt2,
            args,
            train_dataset=train_dataset,
            eval_dataset={"first": first_dataset, "second": second_dataset},
        )
        # Mocking the prepare method to avoid the dataloader changing with each call to get_eval_dataloader
        trainer.accelerator.prepare = lambda x: x

        first_dataloader = trainer.get_eval_dataloader("first")
        first_dataloader_repeated = trainer.get_eval_dataloader("first")
        second_dataloader = trainer.get_eval_dataloader("second")
        second_dataloader_repeated = trainer.get_eval_dataloader("second")

        self.assertEqual(first_dataset, first_dataloader.dataset)
        self.assertEqual(first_dataloader.dataset, first_dataloader_repeated.dataset)
        self.assertEqual(second_dataset, second_dataloader.dataset)
        self.assertEqual(second_dataloader.dataset, second_dataloader_repeated.dataset)
        self.assertEqual(first_dataloader, first_dataloader_repeated)
        self.assertEqual(second_dataloader, second_dataloader_repeated)


# ---------------------------------------------------------------------------
# Label smoothing tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerLabelSmoothingTest(unittest.TestCase):
    """Tests for label smoothing and its interaction with multi-label classification."""

    def test_label_smoothing(self):
        epsilon = 0.1
        num_labels = 12
        random_logits = torch.randn(4, 5, num_labels)
        random_labels = torch.randint(0, num_labels, (4, 5))
        loss = nn.functional.cross_entropy(random_logits.view(-1, num_labels), random_labels.view(-1))
        model_output = SequenceClassifierOutput(logits=random_logits)
        label_smoothed_loss = LabelSmoother(0.1)(model_output, random_labels)
        log_probs = -nn.functional.log_softmax(random_logits, dim=-1)
        expected_loss = (1 - epsilon) * loss + epsilon * log_probs.mean()
        torch.testing.assert_close(label_smoothed_loss, expected_loss)

        # With a few -100 labels
        random_labels[0, 1] = -100
        random_labels[2, 1] = -100
        random_labels[2, 3] = -100

        loss = nn.functional.cross_entropy(random_logits.view(-1, num_labels), random_labels.view(-1))
        model_output = SequenceClassifierOutput(logits=random_logits)
        label_smoothed_loss = LabelSmoother(0.1)(model_output, random_labels)
        log_probs = -nn.functional.log_softmax(random_logits, dim=-1)
        # Mask the log probs with the -100 labels
        log_probs[0, 1] = 0.0
        log_probs[2, 1] = 0.0
        log_probs[2, 3] = 0.0
        expected_loss = (1 - epsilon) * loss + epsilon * log_probs.sum() / (num_labels * 17)
        torch.testing.assert_close(label_smoothed_loss, expected_loss)

    def test_label_smoothing_multi_label_incompatibility(self):
        """Test that Trainer warns and disables label smoothing for multi-label classification"""

        # Mock model config with multi-label classification
        class MockConfig:
            problem_type = "multi_label_classification"

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockConfig()
                self.linear = nn.Linear(10, 3)

            def forward(self, **kwargs):
                return {"logits": torch.randn(2, 3)}

        model = MockModel()

        # Create training args with label smoothing
        training_args = TrainingArguments(
            output_dir="./test-trainer",
            label_smoothing_factor=0.1,
            per_device_train_batch_size=2,
            num_train_epochs=1,
        )

        # Should warn and disable label smoothing
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trainer = Trainer(model=model, args=training_args)

            # Check warning was issued
            self.assertEqual(len(w), 1)
            self.assertIn("Label smoothing is not compatible with multi-label classification", str(w[0].message))

            # Check label_smoother was disabled
            self.assertIsNone(trainer.label_smoother)


# ---------------------------------------------------------------------------
# Sampler and sharding tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerSamplerTest(unittest.TestCase):
    """Tests for length-grouped samplers, distributed samplers, iterable dataset sharding, and shard samplers."""

    def test_group_by_length(self):
        # Get some inputs of random lengths
        lengths = torch.randint(0, 25, (100,)).tolist()
        # Put one bigger than the others to check it ends up in first position
        lengths[32] = 50

        indices = list(LengthGroupedSampler(4, lengths=lengths))
        # The biggest element should be first
        self.assertEqual(lengths[indices[0]], 50)
        # The indices should be a permutation of range(100)
        self.assertEqual(sorted(indices), list(range(100)))

    def test_group_by_length_with_dict(self):
        # Get some inputs of random lengths
        data = []
        for _ in range(6):
            input_ids = torch.randint(0, 25, (100,)).tolist()
            data.append({"input_ids": input_ids})
        # Put one bigger than the others to check it ends up in first position
        data[3]["input_ids"] = torch.randint(0, 25, (105,)).tolist()

        indices = list(LengthGroupedSampler(4, dataset=data))
        # The biggest element should be first
        self.assertEqual(len(data[indices[0]]["input_ids"]), 105)
        # The indices should be a permutation of range(6)
        self.assertEqual(sorted(indices), list(range(6)))

    def test_group_by_length_with_batch_encoding(self):
        # Get some inputs of random lengths
        data = []
        for _ in range(6):
            input_ids = torch.randint(0, 25, (100,)).tolist()
            data.append(BatchEncoding({"input_ids": input_ids}))
        # Put one bigger than the others to check it ends up in first position
        data[3]["input_ids"] = torch.randint(0, 25, (105,)).tolist()

        indices = list(LengthGroupedSampler(4, dataset=data))
        # The biggest element should be first
        self.assertEqual(len(data[indices[0]]["input_ids"]), 105)
        # The indices should be a permutation of range(6)
        self.assertEqual(sorted(indices), list(range(6)))

    def test_distributed_length_grouped(self):
        # Get some inputs of random lengths
        lengths = torch.randint(0, 25, (100,)).tolist()
        # Put one bigger than the others to check it ends up in first position
        lengths[32] = 50

        indices_process_0 = list(DistributedLengthGroupedSampler(4, num_replicas=2, rank=0, lengths=lengths))
        indices_process_1 = list(DistributedLengthGroupedSampler(4, num_replicas=2, rank=1, lengths=lengths))
        # The biggest element should be first
        self.assertEqual(lengths[indices_process_0[0]], 50)
        # The indices should be a permutation of range(100)
        self.assertEqual(sorted(indices_process_0 + indices_process_1), list(range(100)))

    def test_distributed_sampler_with_loop(self):
        batch_size = 16
        for length in [23, 64, 123]:
            dataset = list(range(length))
            shard1 = DistributedSamplerWithLoop(dataset, batch_size, num_replicas=2, rank=0)
            shard2 = DistributedSamplerWithLoop(dataset, batch_size, num_replicas=2, rank=1)

            # Set seeds
            shard1.set_epoch(0)
            shard2.set_epoch(0)

            # Sample
            samples1 = list(shard1)
            samples2 = list(shard2)

            self.assertTrue(len(samples1) % batch_size == 0)
            self.assertTrue(len(samples2) % batch_size == 0)

            total = []
            for sample1, sample2 in zip(samples1, samples2):
                total += [sample1, sample2]

            self.assertEqual(set(total[:length]), set(dataset))
            self.assertEqual(set(total[length:]), set(total[: (len(total) - length)]))

    def check_iterable_dataset_shard(self, dataset, batch_size, drop_last, num_processes=2, epoch=0):
        # Set the seed for the base dataset to get the proper reference.
        dataset.generator.manual_seed(epoch)
        reference = list(dataset)

        shards = [
            IterableDatasetShard(
                dataset, batch_size=batch_size, drop_last=drop_last, num_processes=num_processes, process_index=i
            )
            for i in range(num_processes)
        ]
        for shard in shards:
            shard.set_epoch(epoch)
        shard_lists = [list(shard) for shard in shards]

        for shard in shard_lists:
            # All shards have a number of samples that is a round multiple of batch size
            self.assertTrue(len(shard) % batch_size == 0)
            # All shards have the same number of samples
            self.assertEqual(len(shard), len(shard_lists[0]))

        for shard in shards:
            # All shards know the total number of samples
            self.assertEqual(shard.num_examples, len(reference))

        observed = []
        for idx in range(0, len(shard_lists[0]), batch_size):
            for shard in shard_lists:
                observed += shard[idx : idx + batch_size]

        # If drop_last is False we loop through samples at the beginning to have a size that is a round multiple of
        # batch_size
        if not drop_last:
            while len(reference) < len(observed):
                reference += reference
        self.assertListEqual(observed, reference[: len(observed)])

        # Check equivalence between IterableDataset and ShardSampler
        dataset.generator.manual_seed(epoch)
        reference = list(dataset)

        sampler_shards = [
            ShardSampler(
                reference, batch_size=batch_size, drop_last=drop_last, num_processes=num_processes, process_index=i
            )
            for i in range(num_processes)
        ]
        for shard, sampler_shard in zip(shard_lists, sampler_shards):
            self.assertListEqual(shard, list(sampler_shard))

    def test_iterable_dataset_shard(self):
        dataset = RandomIterableDataset()

        self.check_iterable_dataset_shard(dataset, 4, drop_last=True, num_processes=2, epoch=0)
        self.check_iterable_dataset_shard(dataset, 4, drop_last=False, num_processes=2, epoch=0)

        self.check_iterable_dataset_shard(dataset, 4, drop_last=True, num_processes=3, epoch=42)
        self.check_iterable_dataset_shard(dataset, 4, drop_last=False, num_processes=3, epoch=42)

    def test_iterable_dataset_shard_with_length(self):
        sampler_shards = [
            IterableDatasetShard(list(range(100)), batch_size=4, drop_last=True, num_processes=2, process_index=i)
            for i in range(2)
        ]

        # Build expected shards: each process will have batches of size 4 until there is not enough elements to
        # form two full batches (so we stop at 96 = (100 // (4 * 2)) * 4)
        expected_shards = [[], []]
        current_shard = 0
        for i in range(0, 96, 4):
            expected_shards[current_shard].extend(list(range(i, i + 4)))
            current_shard = 1 - current_shard

        self.assertListEqual([list(shard) for shard in sampler_shards], expected_shards)
        self.assertListEqual([len(shard) for shard in sampler_shards], [len(shard) for shard in expected_shards])

        sampler_shards = [
            IterableDatasetShard(list(range(100)), batch_size=4, drop_last=False, num_processes=2, process_index=i)
            for i in range(2)
        ]
        # When drop_last=False, we get two last full batches by looping back to the beginning.
        expected_shards[0].extend(list(range(96, 100)))
        expected_shards[1].extend(list(range(0, 4)))

        self.assertListEqual([list(shard) for shard in sampler_shards], expected_shards)
        self.assertListEqual([len(shard) for shard in sampler_shards], [len(shard) for shard in expected_shards])

    def check_shard_sampler(self, dataset, batch_size, drop_last, num_processes=2):
        shards = [
            ShardSampler(
                dataset, batch_size=batch_size, drop_last=drop_last, num_processes=num_processes, process_index=i
            )
            for i in range(num_processes)
        ]
        shard_lists = [list(shard) for shard in shards]

        for shard in shard_lists:
            # All shards have a number of samples that is a round multiple of batch size
            self.assertTrue(len(shard) % batch_size == 0)
            # All shards have the same number of samples
            self.assertEqual(len(shard), len(shard_lists[0]))

        observed = []
        for idx in range(0, len(shard_lists[0]), batch_size):
            for shard in shard_lists:
                observed += shard[idx : idx + batch_size]

        # If drop_last is False we loop through samples at the beginning to have a size that is a round multiple of
        # batch_size
        reference = copy.copy(dataset)
        if not drop_last:
            while len(reference) < len(observed):
                reference += reference
        self.assertListEqual(observed, reference[: len(observed)])

    def test_shard_sampler(self):
        for n_elements in [64, 123]:
            dataset = list(range(n_elements))

            self.check_shard_sampler(dataset, 4, drop_last=True, num_processes=2)
            self.check_shard_sampler(dataset, 4, drop_last=False, num_processes=2)

            self.check_shard_sampler(dataset, 4, drop_last=True, num_processes=3)
            self.check_shard_sampler(dataset, 4, drop_last=False, num_processes=3)


# ---------------------------------------------------------------------------
# Batch size finder tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerBatchSizeFinderTest(unittest.TestCase):
    """Tests for the auto batch size finder (find_executable_batch_size)."""

    @require_accelerate
    def test_executable_batch_size(self):
        batch_sizes = []

        @find_executable_batch_size(starting_batch_size=64, auto_find_batch_size=True)
        def mock_training_loop_function(batch_size):
            nonlocal batch_sizes
            batch_sizes.append(batch_size)
            if batch_size > 16:
                raise RuntimeError("CUDA out of memory.")

        mock_training_loop_function()
        self.assertEqual(batch_sizes, [64, 57, 51, 45, 40, 36, 32, 28, 25, 22, 19, 17, 15])

    @require_accelerate
    def test_executable_batch_size_no_search(self):
        batch_sizes = []

        @find_executable_batch_size(starting_batch_size=64, auto_find_batch_size=False)
        def mock_training_loop_function(batch_size):
            nonlocal batch_sizes
            batch_sizes.append(batch_size)

        mock_training_loop_function()
        self.assertEqual(batch_sizes, [64])

    @require_accelerate
    def test_executable_batch_size_with_error(self):
        @find_executable_batch_size(starting_batch_size=64, auto_find_batch_size=False)
        def mock_training_loop_function(batch_size):
            raise RuntimeError("CUDA out of memory.")

        with self.assertRaises(RuntimeError) as cm:
            mock_training_loop_function()
            self.assertEqual("CUDA out of memory", cm.args[0])


# ---------------------------------------------------------------------------
# Data utility tests (parameter names, pad/concat, collators, eval loop container)
# ---------------------------------------------------------------------------


@require_torch
class TrainerDataUtilsTest(unittest.TestCase):
    """Tests for get_parameter_names, pad_and_concatenate, RemoveColumnsCollator, and EvalLoopContainer."""

    def test_get_parameter_names(self):
        model = nn.Sequential(TstLayer(128), nn.ModuleList([TstLayer(128), TstLayer(128)]))
        # fmt: off
        self.assertEqual(
            get_parameter_names(model, [nn.LayerNorm]),
            ['0.linear1.weight', '0.linear1.bias', '0.linear2.weight', '0.linear2.bias', '0.bias', '1.0.linear1.weight', '1.0.linear1.bias', '1.0.linear2.weight', '1.0.linear2.bias', '1.0.bias', '1.1.linear1.weight', '1.1.linear1.bias', '1.1.linear2.weight', '1.1.linear2.bias', '1.1.bias']
        )
        # fmt: on

    def test_get_parameter_names_rmsnorm(self):
        class RMSNorm(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.bias = nn.Parameter(torch.zeros(hidden_size))

        class ModelWithRMSNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 128)
                self.rmsnorm = RMSNorm(128)
                self.bias = nn.Parameter(torch.zeros(128))

        model = ModelWithRMSNorm()
        # Test both type-based and name-based filtering
        decay_parameters = get_parameter_names(model, [], ["bias", "rmsnorm"])

        # Parameters that should be in weight decay
        self.assertIn("linear.weight", decay_parameters)

        # Parameters that should NOT be in weight decay
        self.assertNotIn("linear.bias", decay_parameters)
        self.assertNotIn("rmsnorm.weight", decay_parameters)
        self.assertNotIn("rmsnorm.bias", decay_parameters)
        self.assertNotIn("bias", decay_parameters)

    def test_pad_and_concatenate_with_1d(self):
        """Tests whether pad_and_concatenate works with scalars."""
        array1 = 1.0
        array2 = 2.0
        result = numpy_pad_and_concatenate(array1, array2)
        self.assertTrue(np.array_equal(np.array([1.0, 2.0]), result))

        tensor1 = torch.tensor(1.0)
        tensor2 = torch.tensor(2.0)
        result = torch_pad_and_concatenate(tensor1, tensor2)
        self.assertTrue(torch.equal(result, torch.Tensor([1.0, 2.0])))

    def test_remove_columns_collator(self):
        class MockLogger:
            def __init__(self) -> None:
                self.called = 0

            def info(self, msg):
                self.called += 1
                self.last_msg = msg

        data_batch = [
            {"col1": 1, "col2": 2, "col3": 3},
            {"col1": 1, "col2": 2, "col3": 3},
        ]
        logger = MockLogger()
        remove_columns_collator = RemoveColumnsCollator(
            _default_data_collator, ["col1", "col2"], logger, "model", "training"
        )

        self.assertNotIn("col3", remove_columns_collator(data_batch))
        # check that the logging message is printed out only once
        remove_columns_collator(data_batch)
        remove_columns_collator(data_batch)
        self.assertEqual(logger.called, 1)
        self.assertIn("col3", logger.last_msg)

    def test_eval_loop_container(self):
        batch_1 = [
            torch.ones([8, 5]),
            {"loss": torch.tensor(1.0)},
            (torch.ones([8, 2, 3]), torch.ones([8, 2])),
        ]
        batch_2 = [
            torch.ones([4, 5]),
            {"loss": torch.tensor(2.0)},
            (torch.ones([4, 2, 3]), torch.ones([4, 6])),
        ]

        concat_container = EvalLoopContainer(do_nested_concat=True, padding_index=-100)
        concat_container.add(batch_1)
        concat_container.add(batch_2)
        concat_container.to_cpu_and_numpy()
        arrays = concat_container.get_arrays()

        # Test two nested batches concatenation
        self.assertIsInstance(arrays, list)
        self.assertEqual(len(arrays), 3)
        self.assertIsInstance(arrays[0], np.ndarray)
        self.assertEqual(arrays[0].shape, (12, 5))
        self.assertIsInstance(arrays[1], dict)
        self.assertIsInstance(arrays[1]["loss"], np.ndarray)
        self.assertEqual(arrays[1]["loss"].shape, (2,))
        self.assertTrue(np.allclose(arrays[1]["loss"], np.array([1.0, 2.0])))
        self.assertIsInstance(arrays[2], tuple)
        self.assertEqual(len(arrays[2]), 2)
        self.assertEqual(arrays[2][0].shape, (12, 2, 3))
        self.assertEqual(arrays[2][1].shape, (12, 6))
        # check that first batch padded with padding index -100 after concatenation
        self.assertEqual(arrays[2][1][0][2], -100)

        # Test two batches with no concatenation
        list_container = EvalLoopContainer(do_nested_concat=False)
        list_container.add(batch_1)
        list_container.add(batch_2)
        list_container.to_cpu_and_numpy()
        arrays = list_container.get_arrays()

        self.assertEqual(len(arrays), 2)
        self.assertIsInstance(arrays, list)
        np_batch_1, np_batch_2 = arrays

        self.assertIsInstance(np_batch_1, list)
        self.assertEqual(len(np_batch_1), 3)
        self.assertIsInstance(np_batch_1[0], np.ndarray)
        self.assertIsInstance(np_batch_1[1], dict)
        self.assertIsInstance(np_batch_1[2], tuple)
        self.assertEqual(np_batch_1[0].shape, (8, 5))
        self.assertEqual(np_batch_1[1]["loss"].shape, ())
        self.assertEqual(np_batch_1[2][0].shape, (8, 2, 3))
        self.assertEqual(np_batch_1[2][1].shape, (8, 2))

        self.assertIsInstance(np_batch_2, list)
        self.assertEqual(len(np_batch_2), 3)
        self.assertIsInstance(np_batch_2[0], np.ndarray)
        self.assertIsInstance(np_batch_2[1], dict)
        self.assertIsInstance(np_batch_2[2], tuple)
        self.assertEqual(np_batch_2[0].shape, (4, 5))
        self.assertEqual(np_batch_2[1]["loss"].shape, ())
        self.assertEqual(np_batch_2[2][0].shape, (4, 2, 3))
        self.assertEqual(np_batch_2[2][1].shape, (4, 6))

        # Test no batches
        none_arr = EvalLoopContainer(do_nested_concat=True, padding_index=-100).get_arrays()
        self.assertIsNone(none_arr)

        none_arr = EvalLoopContainer(do_nested_concat=False).get_arrays()
        self.assertIsNone(none_arr)

        # Test one batch
        concat_container = EvalLoopContainer(do_nested_concat=True, padding_index=-100)
        concat_container.add(batch_1)
        arrays = concat_container.get_arrays()
        self.assertIsInstance(arrays, list)
        self.assertEqual(len(arrays), 3)
        self.assertIsInstance(arrays[0], np.ndarray)
        self.assertEqual(arrays[0].shape, (8, 5))
        self.assertIsInstance(arrays[1], dict)
        self.assertIsInstance(arrays[1]["loss"], np.ndarray)
        self.assertEqual(arrays[1]["loss"].shape, ())
        self.assertTrue(np.allclose(arrays[1]["loss"], np.array([1.0])))
        self.assertIsInstance(arrays[2], tuple)
        self.assertEqual(len(arrays[2]), 2)
        self.assertEqual(arrays[2][0].shape, (8, 2, 3))
        self.assertEqual(arrays[2][1].shape, (8, 2))


# ---------------------------------------------------------------------------
# Dynamic shapes and iterable dataset tests
# ---------------------------------------------------------------------------


@require_torch
class TrainerDynamicShapesAndIterableTest(TestCasePlus, TrainerIntegrationCommon):
    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    def test_dynamic_shapes(self):
        eval_dataset = DynamicShapesDataset(batch_size=self.batch_size)
        model = RegressionModel(a=2, b=1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(tmp_dir)
            trainer = Trainer(model, args, eval_dataset=eval_dataset)

            # Check evaluation can run to completion
            _ = trainer.evaluate()

            # Check predictions
            preds = trainer.predict(eval_dataset)
            for expected, seen in zip(eval_dataset.ys, preds.label_ids):
                self.assertTrue(np.array_equal(expected, seen[: expected.shape[0]]))
                self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

            for expected, seen in zip(eval_dataset.xs, preds.predictions):
                self.assertTrue(np.array_equal(2 * expected + 1, seen[: expected.shape[0]]))
                self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

        # Same tests with eval accumulation
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(tmp_dir, eval_accumulation_steps=2)
            trainer = Trainer(model, args, eval_dataset=eval_dataset)

            # Check evaluation can run to completion
            _ = trainer.evaluate()

            # Check predictions
            preds = trainer.predict(eval_dataset)
            for expected, seen in zip(eval_dataset.ys, preds.label_ids):
                self.assertTrue(np.array_equal(expected, seen[: expected.shape[0]]))
                self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

            for expected, seen in zip(eval_dataset.xs, preds.predictions):
                self.assertTrue(np.array_equal(2 * expected + 1, seen[: expected.shape[0]]))
                self.assertTrue(np.all(seen[expected.shape[0] :] == -100))

    def test_training_iterable_dataset(self):
        config = RegressionModelConfig()
        model = RegressionPreTrainedModel(config)
        # Adding one column not used by the model should have no impact
        train_dataset = SampleIterableDataset(label_names=["labels", "extra"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = RegressionTrainingArguments(output_dir=tmp_dir, max_steps=4)
            trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
            trainer.train()
            self.assertEqual(trainer.state.global_step, 4)

            loader = trainer.get_train_dataloader()
            self.assertIsInstance(loader, torch.utils.data.DataLoader)
            self.assertIsInstance(loader.sampler, torch.utils.data.dataloader._InfiniteConstantSampler)

    def test_evaluation_iterable_dataset(self):
        config = RegressionModelConfig(a=1.5, b=2.5)
        model = RegressionPreTrainedModel(config)
        # RegressionPreTrainedModel accepts **kwargs but doesn't actually use num_items_in_batch,
        # so disable the loss scaling that assumes the model handles token-level averaging.
        model.accepts_loss_kwargs = False
        # Adding one column not used by the model should have no impact
        eval_dataset = SampleIterableDataset(label_names=["labels", "extra"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = RegressionTrainingArguments(output_dir=tmp_dir)
            trainer = Trainer(model=model, args=args, eval_dataset=eval_dataset, compute_metrics=AlmostAccuracy())
            results = trainer.evaluate()
            x, y = trainer.eval_dataset.dataset.x, trainer.eval_dataset.dataset.ys[0]
            pred = 1.5 * x + 2.5
            expected_loss = ((pred - y) ** 2).mean()
            self.assertAlmostEqual(results["eval_loss"], expected_loss, places=6)
            expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
            self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

            # With a number of elements not a round multiple of the batch size
            eval_dataset = SampleIterableDataset(length=66)
            results = trainer.evaluate(eval_dataset)

            x, y = eval_dataset.dataset.x, eval_dataset.dataset.ys[0]
            pred = 1.5 * x + 2.5
            expected_loss = ((pred - y) ** 2).mean()
            self.assertAlmostEqual(results["eval_loss"], expected_loss, places=6)
            expected_acc = AlmostAccuracy()((pred, y))["accuracy"]
            self.assertAlmostEqual(results["eval_accuracy"], expected_acc)

    def test_predict_iterable_dataset(self):
        config = RegressionModelConfig(a=1.5, b=2.5)
        model = RegressionPreTrainedModel(config)
        eval_dataset = SampleIterableDataset()

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = RegressionTrainingArguments(output_dir=tmp_dir)
            trainer = Trainer(model=model, args=args, eval_dataset=eval_dataset, compute_metrics=AlmostAccuracy())
            preds = trainer.predict(trainer.eval_dataset).predictions
            x = eval_dataset.dataset.x
            self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))

            # With a number of elements not a round multiple of the batch size
            # Adding one column not used by the model should have no impact
            test_dataset = SampleIterableDataset(length=66, label_names=["labels", "extra"])
            preds = trainer.predict(test_dataset).predictions
            x = test_dataset.dataset.x
            self.assertTrue(np.allclose(preds, 1.5 * x + 2.5))
