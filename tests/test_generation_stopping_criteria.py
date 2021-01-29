import time
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from .test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers.generation_stopping_criteria import MaxLengthCriteria, MaxTimeCriteria, StoppingCriteriaList


@require_torch
class StoppingCriteriaTestCase(unittest.TestCase):
    def _get_tensors(self, length):
        batch_size = 3
        vocab_size = 250

        input_ids = ids_tensor((batch_size, length), vocab_size)
        scores = torch.ones((batch_size, length), device=torch_device, dtype=torch.float) / length
        return input_ids, scores

    def test_list_criteria(self):
        input_ids, scores = self._get_tensors(5)

        criteria = StoppingCriteriaList(
            [
                MaxLengthCriteria(max_length=10),
                MaxTimeCriteria(max_time=0.1),
            ]
        )

        self.assertFalse(criteria(input_ids, scores))

        input_ids, scores = self._get_tensors(10)
        self.assertFalse(criteria(input_ids, scores))

        input_ids, scores = self._get_tensors(11)
        self.assertTrue(criteria(input_ids, scores))

    def test_max_length_criteria(self):
        criteria = MaxLengthCriteria(max_length=10)

        input_ids, scores = self._get_tensors(5)
        self.assertFalse(criteria(input_ids, scores))

        input_ids, scores = self._get_tensors(10)
        self.assertFalse(criteria(input_ids, scores))

        input_ids, scores = self._get_tensors(11)
        self.assertTrue(criteria(input_ids, scores))

    def test_max_time_criteria(self):
        input_ids, scores = self._get_tensors(5)

        criteria = MaxTimeCriteria(max_time=0.1)
        self.assertFalse(criteria(input_ids, scores))

        criteria = MaxTimeCriteria(max_time=0.1, initial_timestamp=time.time() - 0.2)
        self.assertTrue(criteria(input_ids, scores))
