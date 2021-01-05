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
    from transformers.trainer_pt_utils import DistributedTensorGatherer, LabelSmoother


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
        model_output = SequenceClassifierOutput(loss=loss, logits=random_logits)
        label_smoothed_loss = LabelSmoother(0.1)(model_output, random_labels)
        log_probs = -torch.nn.functional.log_softmax(random_logits, dim=-1)
        expected_loss = (1 - epsilon) * loss + epsilon * log_probs.mean()
        self.assertTrue(torch.allclose(label_smoothed_loss, expected_loss))

        # With a few -100 labels
        random_labels[0, 1] = -100
        random_labels[2, 1] = -100
        random_labels[2, 3] = -100

        loss = torch.nn.functional.cross_entropy(random_logits.view(-1, num_labels), random_labels.view(-1))
        model_output = SequenceClassifierOutput(loss=loss, logits=random_logits)
        label_smoothed_loss = LabelSmoother(0.1)(model_output, random_labels)
        log_probs = -torch.nn.functional.log_softmax(random_logits, dim=-1)
        # Mask the log probs with the -100 labels
        log_probs[0, 1] = 0.0
        log_probs[2, 1] = 0.0
        log_probs[2, 3] = 0.0
        expected_loss = (1 - epsilon) * loss + epsilon * log_probs.sum() / (num_labels * 17)
        self.assertTrue(torch.allclose(label_smoothed_loss, expected_loss))
