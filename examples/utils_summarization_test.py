# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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
import torch

from utils_summarization import (
    compute_token_type_ids,
    fit_to_block_size,
    build_mask,
    build_lm_labels,
    process_story,
)


class SummarizationDataProcessingTest(unittest.TestCase):
    def setUp(self):
        self.block_size = 10

    def test_fit_to_block_sequence_too_small(self):
        """ Pad the sequence with 0 if the sequence is smaller than the block size."""
        sequence = [1, 2, 3, 4]
        expected_output = [1, 2, 3, 4, 0, 0, 0, 0, 0, 0]
        self.assertEqual(
            fit_to_block_size(sequence, self.block_size, 0), expected_output
        )

    def test_fit_to_block_sequence_fit_exactly(self):
        """ Do nothing if the sequence is the right size. """
        sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected_output = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(
            fit_to_block_size(sequence, self.block_size, 0), expected_output
        )

    def test_fit_to_block_sequence_too_big(self):
        """ Truncate the sequence if it is too long. """
        sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        expected_output = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(
            fit_to_block_size(sequence, self.block_size, 0), expected_output
        )

    def test_process_story_no_highlights(self):
        """ Processing a story with no highlights returns an empty list for the summary.
        """
        raw_story = """It was the year of Our Lord one thousand seven hundred and
        seventy-five.\n\nSpiritual revelations were conceded to England at that
        favoured period, as at this."""
        _, summary_lines = process_story(raw_story)
        self.assertEqual(summary_lines, [])

    def test_process_empty_story(self):
        """ An empty story returns an empty collection of lines.
        """
        raw_story = ""
        story_lines, summary_lines = process_story(raw_story)
        self.assertEqual(story_lines, [])
        self.assertEqual(summary_lines, [])

    def test_process_story_with_missing_period(self):
        raw_story = (
            "It was the year of Our Lord one thousand seven hundred and "
            "seventy-five\n\nSpiritual revelations were conceded to England "
            "at that favoured period, as at this.\n@highlight\n\nIt was the best of times"
        )
        story_lines, summary_lines = process_story(raw_story)

        expected_story_lines = [
            "It was the year of Our Lord one thousand seven hundred and seventy-five.",
            "Spiritual revelations were conceded to England at that favoured period, as at this.",
        ]
        self.assertEqual(expected_story_lines, story_lines)

        expected_summary_lines = ["It was the best of times."]
        self.assertEqual(expected_summary_lines, summary_lines)

    def test_build_lm_labels_no_padding(self):
        sequence = torch.tensor([1, 2, 3, 4])
        expected = sequence
        np.testing.assert_array_equal(
            build_lm_labels(sequence, 0).numpy(), expected.numpy()
        )

    def test_build_lm_labels(self):
        sequence = torch.tensor([1, 2, 3, 4, 0, 0, 0])
        expected = torch.tensor([1, 2, 3, 4, -1, -1, -1])
        np.testing.assert_array_equal(
            build_lm_labels(sequence, 0).numpy(), expected.numpy()
        )

    def test_build_mask_no_padding(self):
        sequence = torch.tensor([1, 2, 3, 4])
        expected = torch.tensor([1, 1, 1, 1])
        np.testing.assert_array_equal(build_mask(sequence, 0).numpy(), expected.numpy())

    def test_build_mask(self):
        sequence = torch.tensor([1, 2, 3, 4, 23, 23, 23])
        expected = torch.tensor([1, 1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(
            build_mask(sequence, 23).numpy(), expected.numpy()
        )

    def test_build_mask_with_padding_equal_to_one(self):
        sequence = torch.tensor([8, 2, 3, 4, 1, 1, 1])
        expected = torch.tensor([1, 1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(build_mask(sequence, 1).numpy(), expected.numpy())

    def test_compute_token_type_ids(self):
        separator = 101
        batch = torch.tensor(
            [[1, 2, 3, 4, 5, 6], [1, 2, 3, 101, 5, 6], [1, 101, 3, 4, 101, 6]]
        )
        expected = torch.tensor(
            [[0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1], [0, 1, 1, 1, 0, 0]]
        )

        result = compute_token_type_ids(batch, separator)
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
