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

    from transformers.generation_beam_constraints import ConjunctiveDisjunctiveConstraint, DisjunctiveConstraint


@require_torch
class ConstraintTest(unittest.TestCase):
    def test_dc_input_types(self):
        # For consistency across different places the DisjunctiveConstraint is called,
        # dc.token_ids is a list of integers. It is also initialized only by integers.

        cset = [[1, 2, 4], [1, 2, 3, 4]]
        dc = DisjunctiveConstraint(cset)
        self.assertTrue(isinstance(dc.token_ids, list))

        with self.assertRaises(ValueError):
            DisjunctiveConstraint(torch.LongTensor([[1, 2, 4], [1, 2, 3]]))

        with self.assertRaises(ValueError):
            DisjunctiveConstraint([torch.LongTensor([1, 2, 4]), torch.LongTensor([1, 2, 3, 4, 5])])

    def test_dc_check_illegal_input(self):
        # We can't have constraints that are non-suffix complete subsets of another. This leads to a preverse
        # interpretation of "constraint fulfillment": does generating [1,2,3] fulfill the constraint?
        # It would mean that it generated [1,2] which fulfills it, but it's in the middle of potentially
        # fulfilling [1,2,3,4]. If we believe that [1,2,3] does fulfill the constraint, then the algorithm
        # will necessarily never reach [1,2,3,4], giving users a false sense of control (better to just not allow it).
        cset = [[1, 2], [1, 2, 3, 4]]

        with self.assertRaises(ValueError):
            DisjunctiveConstraint(cset)  # fails here

        cset = [[2, 3], [1, 2, 3, 4]]

        with self.assertRaises(ValueError):
            DisjunctiveConstraint(cset)  # fails here

        cset = [[3, 4], [1, 2, 3, 4]]

        DisjunctiveConstraint(cset)  # succeeds here

    def test_dc_example_progression_and_copy(self):
        cset = [[1, 2, 3], [1, 2, 4]]

        dc = DisjunctiveConstraint(cset)

        self.assertTrue(dc.does_advance(1))
        stepped, completed, reset = dc.update(1)
        desired = stepped is True and completed is False and reset is False
        self.assertTrue(desired)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1])
        self.assertTrue(dc.advance() == [2])

        self.assertTrue(dc.does_advance(2))
        stepped, completed, reset = dc.update(2)
        desired = stepped is True and completed is False and reset is False
        self.assertTrue(desired)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1, 2])
        self.assertTrue(dc.advance() == [3, 4])

        self.assertTrue(dc.remaining() == 1)
        dc_copy = dc.copy()
        self.assertTrue(dc_copy.remaining() == 3)
        dc_copy = dc.copy(True)
        self.assertTrue(dc_copy.remaining() == 1)

        self.assertTrue(not dc_copy.does_advance(5))
        stepped, completed, reset = dc_copy.update(5)
        desired = stepped is False and completed is False and reset is True
        self.assertTrue(desired)
        self.assertTrue(not dc_copy.completed)
        self.assertTrue(dc_copy.current_seq == [])  # Reset!
        self.assertTrue(dc_copy.advance() == [1])

        self.assertTrue(dc.does_advance(3))
        stepped, completed, reset = dc.update(3)
        desired = stepped is True and completed is True and reset is False
        self.assertTrue(desired)
        self.assertTrue(dc.completed)  # Completed!
        self.assertTrue(dc.current_seq == [1, 2, 3])
        self.assertTrue(dc.advance() is None)

    def test_dc_example_progression_unequal_three_mid_and_reset(self):
        cset = [[1, 2, 3], [1, 2, 4, 5], [1, 2, 5]]

        dc = DisjunctiveConstraint(cset)

        self.assertTrue(dc.does_advance(1))
        stepped, completed, reset = dc.update(1)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1])
        self.assertTrue(dc.advance() == [2])

        self.assertTrue(dc.does_advance(2))
        stepped, completed, reset = dc.update(2)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1, 2])
        self.assertTrue(dc.advance() == [3, 4, 5])

        self.assertTrue(dc.does_advance(4))
        stepped, completed, reset = dc.update(4)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1, 2, 4])
        self.assertTrue(dc.advance() == [5])

        self.assertTrue(dc.does_advance(5))
        stepped, completed, reset = dc.update(5)
        self.assertTrue(dc.completed)  # Completed!
        self.assertTrue(dc.current_seq == [1, 2, 4, 5])
        self.assertTrue(dc.advance() is None)

        dc.reset()

        self.assertTrue(dc.does_advance(1))
        stepped, completed, reset = dc.update(1)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.remaining() == 3)
        self.assertTrue(dc.current_seq == [1])
        self.assertTrue(dc.advance() == [2])

        self.assertTrue(dc.does_advance(2))
        stepped, completed, reset = dc.update(2)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.remaining() == 2)
        self.assertTrue(dc.current_seq == [1, 2])
        self.assertTrue(dc.advance() == [3, 4, 5])

        self.assertTrue(dc.does_advance(5))
        stepped, completed, reset = dc.update(5)
        self.assertTrue(dc.completed)  # Completed!
        self.assertTrue(dc.remaining() == 0)
        self.assertTrue(dc.current_seq == [1, 2, 5])
        self.assertTrue(dc.advance() is None)

    def test_dc_example_progression_mid_overlap_two(self):
        cset = [[1, 2, 3], [2, 4]]

        dc = DisjunctiveConstraint(cset)

        self.assertTrue(dc.does_advance(1))
        stepped, completed, reset = dc.update(1)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1])
        self.assertTrue(dc.advance() == [2])

        self.assertTrue(dc.does_advance(2))
        stepped, completed, reset = dc.update(2)
        self.assertTrue(not dc.completed)
        self.assertTrue(dc.current_seq == [1, 2])
        self.assertTrue(dc.advance() == [3])

        self.assertTrue(dc.does_advance(4))
        stepped, completed, reset = dc.update(4)
        self.assertTrue(dc.completed)  # Completed!
        self.assertTrue(dc.current_seq == [1, 2, 4])
        self.assertTrue(dc.advance() is None)

    def test_cdc_input(self):
        cset = [[], [[]], [[], []], [1], [[1]], [[1], []], [[1], [1]], [[1], [1], []], [[1], [1, 2], [0, 1, 2]]]

        cdc = ConjunctiveDisjunctiveConstraint(cset)  # succeeds here

        self.assertTrue(cdc.remaining() == 7)
        self.assertTrue(len(cdc.ac_automaton.trie) == 4)
        self.assertTrue(cdc.ac_automaton.conj_count[1] == 5)
        self.assertTrue(cdc.ac_automaton.disj_set[1] == {0})
        self.assertTrue(cdc.ac_automaton.conj_count[3] == 0)
        self.assertTrue(cdc.ac_automaton.disj_set[3] == {0})

    def test_cdc_example_progression_and_copy(self):
        cset = [[[1, 2, 3], [1, 2, 4]]]

        cdc = ConjunctiveDisjunctiveConstraint(cset)

        self.assertTrue(cdc.does_advance(1))
        stepped, completed, reset = cdc.update(1)
        desired = stepped is True and completed is False and reset is False
        self.assertTrue(desired)
        self.assertTrue(cdc.advance() == [2])

        self.assertTrue(cdc.does_advance(2))
        stepped, completed, reset = cdc.update(2)
        desired = stepped is True and completed is False and reset is False
        self.assertTrue(desired)
        self.assertTrue(sorted(cdc.advance()) == [3, 4])

        self.assertTrue(cdc.remaining() == 1)
        cdc_copy = cdc.copy()
        self.assertTrue(cdc_copy.remaining() == 3)
        cdc_copy = cdc.copy(True)
        self.assertTrue(cdc_copy.remaining() == 1)

        self.assertTrue(not cdc_copy.does_advance(5))
        stepped, completed, reset = cdc_copy.update(5)
        desired = stepped is False and completed is False and reset is True
        self.assertTrue(desired)  # Reset!
        self.assertTrue(cdc_copy.advance() == [1])

        self.assertTrue(cdc.does_advance(3))
        stepped, completed, reset = cdc.update(3)
        desired = stepped is True and completed is True and reset is False
        self.assertTrue(desired)  # Completed!
        self.assertTrue(cdc.advance() is None)

    def test_cdc_example_progression_unequal_three_mid_and_reset(self):
        cset = [[[1, 2, 3], [1, 2, 4, 5], [1, 2, 5]]]

        cdc = ConjunctiveDisjunctiveConstraint(cset)

        self.assertTrue(cdc.does_advance(1))
        stepped, completed, reset = cdc.update(1)
        self.assertTrue(not completed)
        self.assertTrue(cdc.advance() == [2])

        self.assertTrue(cdc.does_advance(2))
        stepped, completed, reset = cdc.update(2)
        self.assertTrue(not completed)
        self.assertTrue(sorted(cdc.advance()) == [3, 4, 5])

        self.assertTrue(cdc.does_advance(4))
        stepped, completed, reset = cdc.update(4)
        self.assertTrue(not completed)
        self.assertTrue(cdc.advance() == [5])

        self.assertTrue(cdc.does_advance(5))
        stepped, completed, reset = cdc.update(5)
        self.assertTrue(completed)  # Completed!
        self.assertTrue(cdc.advance() is None)

        cdc.reset()

        self.assertTrue(cdc.does_advance(1))
        stepped, completed, reset = cdc.update(1)
        self.assertTrue(not completed)
        self.assertTrue(cdc.remaining() == 3)
        self.assertTrue(cdc.advance() == [2])

        self.assertTrue(cdc.does_advance(2))
        stepped, completed, reset = cdc.update(2)
        self.assertTrue(not completed)
        self.assertTrue(cdc.remaining() == 2)
        self.assertTrue(sorted(cdc.advance()) == [3, 4, 5])

        self.assertTrue(cdc.does_advance(5))
        stepped, completed, reset = cdc.update(5)
        self.assertTrue(completed)  # Completed!
        self.assertTrue(cdc.remaining() == 0)
        self.assertTrue(cdc.advance() is None)

    def test_cdc_example_progression_mid_overlap_two(self):
        cset = [[[1, 2, 3], [2, 4]]]

        cdc = ConjunctiveDisjunctiveConstraint(cset)

        self.assertTrue(cdc.does_advance(1))
        stepped, completed, reset = cdc.update(1)
        self.assertTrue(not completed)
        self.assertTrue(cdc.advance() == [2])

        self.assertTrue(cdc.does_advance(2))
        stepped, completed, reset = cdc.update(2)
        self.assertTrue(not completed)
        self.assertTrue(cdc.advance() == [3])

        self.assertTrue(cdc.does_advance(4))
        stepped, completed, reset = cdc.update(4)
        self.assertTrue(completed)  # Completed!
        self.assertTrue(cdc.advance() is None)

    def test_cdc_example_progression_loop_three(self):
        cset = [[[1], [2]], [[2], [3]], [[3], [1]]]

        cdc = ConjunctiveDisjunctiveConstraint(cset)

        self.assertTrue(cdc.does_advance(1))
        stepped, completed, reset = cdc.update(1)
        self.assertTrue(not completed)
        self.assertTrue(cdc.remaining() == 2)
        self.assertTrue(sorted(cdc.advance()) == [1, 2, 3])

        cdc_copy = cdc.copy(True)
        self.assertTrue(cdc_copy.does_advance(1))
        stepped, completed, reset = cdc_copy.update(1)
        self.assertTrue(not completed)
        self.assertTrue(cdc_copy.remaining() == 1)
        self.assertTrue(cdc_copy.advance() == [2, 3])

        self.assertTrue(cdc_copy.does_advance(2))
        stepped, completed, reset = cdc_copy.update(2)
        self.assertTrue(completed)  # Completed!
        self.assertTrue(cdc_copy.remaining() == 0)
        self.assertTrue(cdc_copy.advance() is None)

        self.assertTrue(cdc.does_advance(2))
        stepped, completed, reset = cdc.update(2)
        self.assertTrue(not completed)
        self.assertTrue(cdc.remaining() == 1)
        self.assertTrue(sorted(cdc.advance()) == [1, 2, 3])

        cdc_copy = cdc.copy(True)
        self.assertTrue(cdc_copy.does_advance(1))
        stepped, completed, reset = cdc_copy.update(1)
        self.assertTrue(completed)  # Completed!
        self.assertTrue(cdc_copy.remaining() == 0)
        self.assertTrue(cdc_copy.advance() is None)

        self.assertTrue(cdc.does_advance(3))
        stepped, completed, reset = cdc.update(3)
        self.assertTrue(completed)  # Completed!
        self.assertTrue(cdc.remaining() == 0)
        self.assertTrue(cdc.advance() is None)

    def test_cdc_example_progression_overlap_four(self):
        cset = [[1, 2, 3, 4, 5], [1], [3, 4], [4, 1]]

        cdc = ConjunctiveDisjunctiveConstraint(cset)

        self.assertTrue(cdc.does_advance(1))
        stepped, completed, reset = cdc.update(1)
        self.assertTrue(not completed)
        self.assertTrue(cdc.remaining() == 9)
        self.assertTrue(cdc.advance() == [2])

        self.assertTrue(cdc.does_advance(2))
        stepped, completed, reset = cdc.update(2)
        self.assertTrue(not completed)
        self.assertTrue(cdc.remaining() == 8)
        self.assertTrue(cdc.advance() == [3])

        self.assertTrue(cdc.does_advance(3))
        stepped, completed, reset = cdc.update(3)
        self.assertTrue(cdc.remaining() == 7)
        self.assertTrue(cdc.advance() == [4])

        self.assertTrue(cdc.does_advance(4))
        stepped, completed, reset = cdc.update(4)
        self.assertTrue(not completed)
        self.assertTrue(cdc.remaining() == 6)
        self.assertTrue(cdc.advance() == [5])

        self.assertTrue(cdc.does_advance(1))
        stepped, completed, reset = cdc.update(1)
        desired = stepped is True and completed is False and reset is False
        self.assertTrue(desired)
        self.assertTrue(not completed)
        self.assertTrue(cdc.remaining() == 4)
        self.assertTrue(cdc.advance() == [2])

        self.assertTrue(cdc.does_advance(2))
        stepped, completed, reset = cdc.update(2)
        self.assertTrue(not completed)
        self.assertTrue(cdc.remaining() == 3)
        self.assertTrue(cdc.advance() == [3])

        self.assertTrue(cdc.does_advance(3))
        stepped, completed, reset = cdc.update(3)
        self.assertTrue(not completed)
        self.assertTrue(cdc.remaining() == 2)
        self.assertTrue(cdc.advance() == [4])

        self.assertTrue(cdc.does_advance(4))
        stepped, completed, reset = cdc.update(4)
        self.assertTrue(not completed)
        self.assertTrue(cdc.remaining() == 1)
        self.assertTrue(cdc.advance() == [5])

        self.assertTrue(cdc.does_advance(5))
        stepped, completed, reset = cdc.update(5)
        self.assertTrue(completed)  # Completed!
        self.assertTrue(cdc.remaining() == 0)
        self.assertTrue(cdc.advance() is None)

    def test_cdc_example_progression_ambiguous_eight(self):
        cset = [[1], [[1], [2]], [[1], [2], [3]], [[3], [4]], [[3], [5]], [[3], [6], [7]], [[7], [8]], [4]]

        cdc = ConjunctiveDisjunctiveConstraint(cset)

        self.assertTrue(cdc.does_advance(1))
        stepped, completed, reset = cdc.update(1)
        self.assertTrue(not completed)
        self.assertTrue(cdc.remaining() == 7)
        self.assertTrue(sorted(cdc.advance()) == [1, 2, 3, 4, 5, 6, 7, 8])

        self.assertTrue(cdc.does_advance(1))
        stepped, completed, reset = cdc.update(1)
        desired = stepped is True and completed is False and reset is False
        self.assertTrue(desired)
        self.assertTrue(cdc.remaining() == 6)
        self.assertTrue(sorted(cdc.advance()) == [1, 2, 3, 4, 5, 6, 7, 8])

        self.assertTrue(cdc.does_advance(2))
        stepped, completed, reset = cdc.update(2)
        self.assertTrue(not completed)
        self.assertTrue(cdc.remaining() == 5)
        self.assertTrue(sorted(cdc.advance()) == [3, 4, 5, 6, 7, 8])

        self.assertTrue(cdc.does_advance(3))
        stepped, completed, reset = cdc.update(3)
        self.assertTrue(not completed)
        self.assertTrue(cdc.remaining() == 4)
        self.assertTrue(sorted(cdc.advance()) == [3, 4, 5, 6, 7, 8])

        self.assertTrue(cdc.does_advance(3))
        stepped, completed, reset = cdc.update(3)
        self.assertTrue(not completed)
        self.assertTrue(cdc.remaining() == 3)
        self.assertTrue(sorted(cdc.advance()) == [3, 4, 5, 6, 7, 8])

        self.assertTrue(cdc.does_advance(6))
        stepped, completed, reset = cdc.update(6)
        self.assertTrue(not completed)
        self.assertTrue(cdc.remaining() == 2)
        self.assertTrue(sorted(cdc.advance()) == [4, 7, 8])

        self.assertTrue(cdc.does_advance(4))
        stepped, completed, reset = cdc.update(4)
        self.assertTrue(not completed)
        self.assertTrue(cdc.remaining() == 1)
        self.assertTrue(sorted(cdc.advance()) == [7, 8])

        self.assertTrue(cdc.does_advance(7))
        stepped, completed, reset = cdc.update(7)
        self.assertTrue(completed)  # Completed!
        self.assertTrue(cdc.remaining() == 0)
        self.assertTrue(cdc.advance() is None)
