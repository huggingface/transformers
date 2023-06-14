# coding=utf-8
# Copyright 2020 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
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

    from transformers.generation import DisjunctiveConstraint


@require_torch
class ConstraintTest(unittest.TestCase):
    def test_input_types(self):
        # For consistency across different places the DisjunctiveConstraint is called,
        # dc.token_ids is a list of integers. It is also initialized only by integers.

        cset = [[1, 2, 4], [1, 2, 3, 4]]
        dc = DisjunctiveConstraint(cset)
        self.assertTrue(isinstance(dc.token_ids, list))

        with self.assertRaises(ValueError):
            DisjunctiveConstraint(torch.LongTensor([[1, 2, 4], [1, 2, 3]]))

        with self.assertRaises(ValueError):
            DisjunctiveConstraint([torch.LongTensor([1, 2, 4]), torch.LongTensor([1, 2, 3, 4, 5])])

    def test_check_illegal_input(self):
        # We can't have constraints that are complete subsets of another. This leads to a preverse
        # interpretation of "constraint fulfillment": does generating [1,2,3] fulfill the constraint?
        # It would mean that it generated [1,2] which fulfills it, but it's in the middle of potentially
        # fulfilling [1,2,3,4]. If we believe that [1,2,3] does fulfill the constraint, then the algorithm
        # will necessarily never reach [1,2,3,4], giving users a false sense of control (better to just not allow it).
        cset = [[1, 2], [1, 2, 3, 4]]

        with self.assertRaises(ValueError):
            DisjunctiveConstraint(cset)  # fails here

    def test_example_progression(self):
        cset = [[1, 2, 3], [1, 2, 4]]

        dc = DisjunctiveConstraint(cset)

        stepped, completed, reset = dc.update(1)
        desired = stepped is True and completed is False and reset is False
        self.assertTrue(desired)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1])

        stepped, completed, reset = dc.update(2)
        desired = stepped is True and completed is False and reset is False
        self.assertTrue(desired)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1, 2])

        stepped, completed, reset = dc.update(3)
        desired = stepped is True and completed is True and reset is False
        self.assertTrue(desired)
        self.assertTrue(dc.completed)  # Completed!
        self.assertTrue(dc.current_seq == [1, 2, 3])

    def test_example_progression_unequal_three_mid_and_reset(self):
        cset = [[1, 2, 3], [1, 2, 4, 5], [1, 2, 5]]

        dc = DisjunctiveConstraint(cset)

        stepped, completed, reset = dc.update(1)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1])

        stepped, completed, reset = dc.update(2)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1, 2])

        stepped, completed, reset = dc.update(4)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1, 2, 4])

        stepped, completed, reset = dc.update(5)
        self.assertTrue(dc.completed)  # Completed!
        self.assertTrue(dc.current_seq == [1, 2, 4, 5])

        dc.reset()

        stepped, completed, reset = dc.update(1)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.remaining() == 3)
        self.assertTrue(dc.current_seq == [1])

        stepped, completed, reset = dc.update(2)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.remaining() == 2)
        self.assertTrue(dc.current_seq == [1, 2])

        stepped, completed, reset = dc.update(5)
        self.assertTrue(dc.completed)  # Completed!
        self.assertTrue(dc.remaining() == 0)
        self.assertTrue(dc.current_seq == [1, 2, 5])
