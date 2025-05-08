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

    from transformers.generation import DisjunctiveConstraint, OrderedConstraint, TemplateConstraint


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
        # We can't have constraints that are complete subsets of another. This leads to a perverse
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


@require_torch
class TemplateConstraintTest(unittest.TestCase):
    def test_initialization(self):
        template = [5, None, 3]
        constraint = TemplateConstraint(template)
        self.assertEqual(constraint.template, template)
        self.assertEqual(constraint.seqlen, 3)
        self.assertEqual(constraint.position, 0)
        self.assertFalse(constraint.completed)

    def test_advance_before_completion(self):
        template = [5, None, 3]
        constraint = TemplateConstraint(template)
        self.assertEqual(constraint.advance(), 5)
        constraint.position = 1
        self.assertIsNone(constraint.advance())
        constraint.position = 2
        self.assertEqual(constraint.advance(), 3)

    def test_advance_after_completion(self):
        template = [5]
        constraint = TemplateConstraint(template)
        constraint.update(5)
        self.assertEqual(constraint.advance(), [])

    def test_does_advance_correct_token(self):
        template = [5, None, 3]
        constraint = TemplateConstraint(template)
        self.assertTrue(constraint.does_advance(5))
        constraint.position = 1
        self.assertTrue(constraint.does_advance(100))  # Any token allowed
        constraint.position = 2
        self.assertTrue(constraint.does_advance(3))
        self.assertFalse(constraint.does_advance(4))

    def test_does_advance_when_completed(self):
        template = [5]
        constraint = TemplateConstraint(template)
        constraint.update(5)
        self.assertFalse(constraint.does_advance(5))

    def test_update_correct_sequence(self):
        template = [5, None, 3]
        constraint = TemplateConstraint(template)
        stepped, completed, reset = constraint.update(5)
        self.assertTrue(stepped)
        self.assertFalse(completed)
        self.assertFalse(reset)
        self.assertEqual(constraint.position, 1)

        stepped, completed, reset = constraint.update(10)  # None allows any token
        self.assertTrue(stepped)
        self.assertFalse(completed)
        self.assertFalse(reset)
        self.assertEqual(constraint.position, 2)

        stepped, completed, reset = constraint.update(3)
        self.assertTrue(stepped)
        self.assertTrue(completed)
        self.assertFalse(reset)
        self.assertEqual(constraint.position, 3)

    def test_update_incorrect_token_resets(self):
        template = [5, 6]
        constraint = TemplateConstraint(template)
        constraint.update(5)
        stepped, completed, reset = constraint.update(7)  # Incorrect
        self.assertFalse(stepped)
        self.assertFalse(completed)
        self.assertTrue(reset)
        self.assertEqual(constraint.position, 0)
        self.assertFalse(constraint.completed)

    def test_reset(self):
        template = [5, 6]
        constraint = TemplateConstraint(template)
        constraint.update(5)
        constraint.reset()
        self.assertEqual(constraint.position, 0)
        self.assertFalse(constraint.completed)

    def test_remaining(self):
        template = [5, None, 3]
        constraint = TemplateConstraint(template)
        self.assertEqual(constraint.remaining(), 3)
        constraint.update(5)
        self.assertEqual(constraint.remaining(), 2)
        constraint.update(10)
        self.assertEqual(constraint.remaining(), 1)
        constraint.update(3)
        self.assertEqual(constraint.remaining(), 0)

    def test_copy_without_state(self):
        template = [5, None]
        original = TemplateConstraint(template)
        original.update(5)
        copied = original.copy(stateful=False)
        self.assertEqual(copied.position, 0)
        self.assertFalse(copied.completed)

    def test_copy_with_state(self):
        template = [5, None]
        original = TemplateConstraint(template)
        original.update(5)
        copied = original.copy(stateful=True)
        self.assertEqual(copied.position, 1)
        self.assertEqual(copied.completed, original.completed)

    def test_all_none_template(self):
        template = [None, None, None]
        constraint = TemplateConstraint(template)
        self.assertTrue(constraint.does_advance(0))
        constraint.update(0)
        self.assertTrue(constraint.does_advance(1))
        constraint.update(1)
        self.assertTrue(constraint.does_advance(2))
        constraint.update(2)
        self.assertTrue(constraint.completed)

    def test_reset_and_retry(self):
        template = [5, 6]
        constraint = TemplateConstraint(template)
        constraint.update(10)  # Incorrect, resets
        constraint.update(5)
        constraint.update(6)
        self.assertTrue(constraint.completed)

    def test_single_token_template(self):
        template = [10]
        constraint = TemplateConstraint(template)
        self.assertTrue(constraint.does_advance(10))
        stepped, completed, reset = constraint.update(10)
        self.assertTrue(stepped)
        self.assertTrue(completed)
        self.assertFalse(reset)

    def test_position_after_reset(self):
        template = [5, 6]
        constraint = TemplateConstraint(template)
        constraint.update(5)
        constraint.update(7)  # Resets
        self.assertEqual(constraint.position, 0)
        constraint.update(5)
        constraint.update(6)
        self.assertTrue(constraint.completed)


@require_torch
class TestOrderedConstraint(unittest.TestCase):
    def test_initialization(self):
        tokens = [5, 2, 3]
        constraint = OrderedConstraint(tokens)
        self.assertEqual(constraint.ordered_token_ids, tokens)
        self.assertEqual(constraint.position, 0)
        self.assertFalse(constraint.completed)
        self.assertEqual(constraint.seqlen, 3)

    def test_advance_before_completion(self):
        tokens = [5, 6, 3]
        constraint = OrderedConstraint(tokens)

        # Position 0
        self.assertEqual(constraint.advance(), 5)
        constraint.position = 1
        self.assertEqual(constraint.advance(), 6)
        constraint.position = 2
        self.assertEqual(constraint.advance(), 3)

    def test_advance_after_completion(self):
        tokens = [5]
        constraint = OrderedConstraint(tokens)
        constraint.update(5)  # Completes the sequence
        self.assertEqual(constraint.advance(), [])

    def test_does_advance_behavior(self):
        tokens = [5, 6, 3]
        constraint = OrderedConstraint(tokens)

        # Position 0
        self.assertTrue(constraint.does_advance(5))
        self.assertFalse(constraint.does_advance(6))

        # Position 1
        constraint.position = 1
        self.assertTrue(constraint.does_advance(6))
        self.assertFalse(constraint.does_advance(5))

        # Position 2
        constraint.position = 2
        self.assertTrue(constraint.does_advance(3))
        self.assertFalse(constraint.does_advance(4))

    def test_update_correct_sequence(self):
        tokens = [5, 6, 3]
        constraint = OrderedConstraint(tokens)

        # First token (5)
        stepped, completed, reset = constraint.update(5)
        self.assertTrue(stepped)
        self.assertFalse(completed)
        self.assertFalse(reset)
        self.assertEqual(constraint.position, 1)

        # Second token (6)
        stepped, completed, reset = constraint.update(6)
        self.assertTrue(stepped)
        self.assertFalse(completed)
        self.assertFalse(reset)
        self.assertEqual(constraint.position, 2)

        # Third token (3)
        stepped, completed, reset = constraint.update(3)
        self.assertTrue(stepped)
        self.assertTrue(completed)
        self.assertFalse(reset)
        self.assertEqual(constraint.position, 3)

    def test_update_incorrect_token_does_not_advance(self):
        tokens = [5, 6]
        constraint = OrderedConstraint(tokens)

        # Correct first token
        constraint.update(5)

        # Incorrect second token
        stepped, completed, reset = constraint.update(7)
        self.assertFalse(stepped)
        self.assertFalse(completed)
        self.assertFalse(reset)  # No reset, position remains at 1
        self.assertEqual(constraint.position, 1)

        # Correct token later
        stepped, completed, _ = constraint.update(6)
        self.assertTrue(stepped)
        self.assertTrue(completed)
        self.assertEqual(constraint.position, 2)

    def test_reset_behavior(self):
        tokens = [5, 6]
        constraint = OrderedConstraint(tokens)
        constraint.update(5)
        constraint.reset()
        self.assertEqual(constraint.position, 0)
        self.assertFalse(constraint.completed)

    def test_remaining_tokens(self):
        tokens = [5, 6, 3]
        constraint = OrderedConstraint(tokens)
        self.assertEqual(constraint.remaining(), 3)
        constraint.update(5)
        self.assertEqual(constraint.remaining(), 2)
        constraint.update(6)
        self.assertEqual(constraint.remaining(), 1)
        constraint.update(3)
        self.assertEqual(constraint.remaining(), 0)

    def test_copy_without_state(self):
        tokens = [5, 6]
        original = OrderedConstraint(tokens)
        original.update(5)
        copied = original.copy(stateful=False)
        self.assertEqual(copied.position, 0)
        self.assertFalse(copied.completed)

    def test_copy_with_state(self):
        tokens = [5, 6]
        original = OrderedConstraint(tokens)
        original.update(5)
        copied = original.copy(stateful=True)
        self.assertEqual(copied.position, 1)
        self.assertEqual(copied.completed, original.completed)

    def test_single_token_completion(self):
        tokens = [10]
        constraint = OrderedConstraint(tokens)
        self.assertTrue(constraint.does_advance(10))
        stepped, completed, _ = constraint.update(10)
        self.assertTrue(stepped)
        self.assertTrue(completed)

    def test_position_overflow(self):
        tokens = [5, 6]
        constraint = OrderedConstraint(tokens)
        constraint.position = 2  # Force beyond sequence length
        self.assertEqual(constraint.advance(), [])
        self.assertTrue(constraint.completed)

    def test_no_progress_on_mismatch(self):
        tokens = [5, 6, 7]
        constraint = OrderedConstraint(tokens)

        # Correct first token
        constraint.update(5)

        # Incorrect second token (should block progress)
        stepped, completed, reset = constraint.update(8)
        self.assertFalse(stepped)
        self.assertFalse(completed)
        self.assertEqual(constraint.position, 1)  # Still at position 1

        # Correct second token later
        stepped, completed, _ = constraint.update(6)
        self.assertTrue(stepped)
        self.assertFalse(completed)
        self.assertEqual(constraint.position, 2)
