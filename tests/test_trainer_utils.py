# coding=utf-8
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

import unittest

import numpy as np

from transformers.file_utils import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers.modeling_outputs import SequenceClassifierOutput
    from transformers.trainer_pt_utils import (
        DistributedLengthGroupedSampler,
        DistributedSamplerWithLoop,
        DistributedTensorGatherer,
        LabelSmoother,
        LengthGroupedSampler,
        SequentialDistributedSampler,
        get_parameter_names,
    )

    class TstLayer(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
            self.ln1 = torch.nn.LayerNorm(hidden_size)
            self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
            self.ln2 = torch.nn.LayerNorm(hidden_size)
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

        def forward(self, x):
            h = self.ln1(torch.nn.functional.relu(self.linear1(x)))
            h = torch.nn.functional.relu(self.linear2(x))
            return self.ln2(x + h + self.bias)


@require_torch
class TrainerUtilsTest(unittest.TestCase):
    def test_distributed_tensor_gatherer(self):
        # Simulate a result with a dataset of size 21, 4 processes and chunks of lengths 2, 3, 1
        world_size = 4
        num_samples = 21
        input_indices = [
            [0, 1, 6, 7, 12, 13, 18, 19],
            [2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 0, 1],
            [5, 11, 17, 2],
        ]

        predictions = np.random.normal(size=(num_samples, 13))
        gatherer = DistributedTensorGatherer(world_size=world_size, num_samples=num_samples)
        for indices in input_indices:
            gatherer.add_arrays(predictions[indices])
        result = gatherer.finalize()
        self.assertTrue(np.array_equal(result, predictions))

        # With nested tensors
        gatherer = DistributedTensorGatherer(world_size=world_size, num_samples=num_samples)
        for indices in input_indices:
            gatherer.add_arrays([predictions[indices], [predictions[indices], predictions[indices]]])
        result = gatherer.finalize()
        self.assertTrue(isinstance(result, list))
        self.assertTrue(len(result), 2)
        self.assertTrue(isinstance(result[1], list))
        self.assertTrue(len(result[1]), 2)
        self.assertTrue(np.array_equal(result[0], predictions))
        self.assertTrue(np.array_equal(result[1][0], predictions))
        self.assertTrue(np.array_equal(result[1][1], predictions))

    def test_label_smoothing(self):
        epsilon = 0.1
        num_labels = 12
        random_logits = torch.randn(4, 5, num_labels)
        random_labels = torch.randint(0, num_labels, (4, 5))
        loss = torch.nn.functional.cross_entropy(random_logits.view(-1, num_labels), random_labels.view(-1))
        model_output = SequenceClassifierOutput(logits=random_logits)
        label_smoothed_loss = LabelSmoother(0.1)(model_output, random_labels)
        log_probs = -torch.nn.functional.log_softmax(random_logits, dim=-1)
        expected_loss = (1 - epsilon) * loss + epsilon * log_probs.mean()
        self.assertTrue(torch.allclose(label_smoothed_loss, expected_loss))

        # With a few -100 labels
        random_labels[0, 1] = -100
        random_labels[2, 1] = -100
        random_labels[2, 3] = -100

        loss = torch.nn.functional.cross_entropy(random_logits.view(-1, num_labels), random_labels.view(-1))
        model_output = SequenceClassifierOutput(logits=random_logits)
        label_smoothed_loss = LabelSmoother(0.1)(model_output, random_labels)
        log_probs = -torch.nn.functional.log_softmax(random_logits, dim=-1)
        # Mask the log probs with the -100 labels
        log_probs[0, 1] = 0.0
        log_probs[2, 1] = 0.0
        log_probs[2, 3] = 0.0
        expected_loss = (1 - epsilon) * loss + epsilon * log_probs.sum() / (num_labels * 17)
        self.assertTrue(torch.allclose(label_smoothed_loss, expected_loss))

    def test_group_by_length(self):
        # Get some inputs of random lengths
        lengths = torch.randint(0, 25, (100,)).tolist()
        # Put one bigger than the others to check it ends up in first position
        lengths[32] = 50

        indices = list(LengthGroupedSampler(lengths, 4, lengths=lengths))
        # The biggest element should be first
        self.assertEqual(lengths[indices[0]], 50)
        # The indices should be a permutation of range(100)
        self.assertEqual(list(sorted(indices)), list(range(100)))

    def test_distributed_length_grouped(self):
        # Get some inputs of random lengths
        lengths = torch.randint(0, 25, (100,)).tolist()
        # Put one bigger than the others to check it ends up in first position
        lengths[32] = 50

        indices_process_0 = list(DistributedLengthGroupedSampler(lengths, 4, 2, 0, lengths=lengths))
        indices_process_1 = list(DistributedLengthGroupedSampler(lengths, 4, 2, 1, lengths=lengths))
        # The biggest element should be first
        self.assertEqual(lengths[indices_process_0[0]], 50)
        # The indices should be a permutation of range(100)
        self.assertEqual(list(sorted(indices_process_0 + indices_process_1)), list(range(100)))

    def test_get_parameter_names(self):
        model = torch.nn.Sequential(TstLayer(128), torch.nn.ModuleList([TstLayer(128), TstLayer(128)]))
        # fmt: off
        self.assertEqual(
            get_parameter_names(model, [torch.nn.LayerNorm]),
            ['0.linear1.weight', '0.linear1.bias', '0.linear2.weight', '0.linear2.bias', '0.bias', '1.0.linear1.weight', '1.0.linear1.bias', '1.0.linear2.weight', '1.0.linear2.bias', '1.0.bias', '1.1.linear1.weight', '1.1.linear1.bias', '1.1.linear2.weight', '1.1.linear2.bias', '1.1.bias']
        )
        # fmt: on

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

    def test_sequential_distributed_sampler(self):
        batch_size = 16
        for length in [23, 64, 123]:
            dataset = list(range(length))
            shard1 = SequentialDistributedSampler(dataset, num_replicas=2, rank=0)
            shard2 = SequentialDistributedSampler(dataset, num_replicas=2, rank=1)

            # Sample
            samples1 = list(shard1)
            samples2 = list(shard2)

            total = samples1 + samples2

            self.assertListEqual(total[:length], dataset)
            self.assertListEqual(total[length:], dataset[: (len(total) - length)])

            # With a batch_size passed
            shard1 = SequentialDistributedSampler(dataset, num_replicas=2, rank=0, batch_size=batch_size)
            shard2 = SequentialDistributedSampler(dataset, num_replicas=2, rank=1, batch_size=batch_size)

            # Sample
            samples1 = list(shard1)
            samples2 = list(shard2)

            self.assertTrue(len(samples1) % batch_size == 0)
            self.assertTrue(len(samples2) % batch_size == 0)

            total = samples1 + samples2

            self.assertListEqual(total[:length], dataset)
            self.assertListEqual(total[length:], dataset[: (len(total) - length)])
