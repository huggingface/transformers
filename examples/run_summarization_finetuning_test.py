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

from run_summarization_finetuning import _fit_to_block_size, process_story


class DataLoaderTest(unittest.TestCase):
    def setUp(self):
        self.block_size = 10

    def test_truncate_sequence_too_small(self):
        """ Pad the sequence with 0 if the sequence is smaller than the block size."""
        sequence = [1, 2, 3, 4]
        expected_output = [1, 2, 3, 4, 0, 0, 0, 0, 0, 0]
        self.assertEqual(_fit_to_block_size(sequence, self.block_size), expected_output)

    def test_truncate_sequence_fit_exactly(self):
        sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected_output = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(_fit_to_block_size(sequence, self.block_size), expected_output)

    def test_truncate_sequence_too_big(self):
        sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        expected_output = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(_fit_to_block_size(sequence, self.block_size), expected_output)

    def test_process_story_no_highlights(self):
        """ Processing a story with no highlights should raise an exception.
        """
        raw_story = """It was the year of Our Lord one thousand seven hundred and
        seventy-five.\n\nSpiritual revelations were conceded to England at that
        favoured period, as at this."""
        _, summary = process_story(raw_story)
        self.assertEqual(summary, [])

    def test_process_empty_story(self):
        """ An empty story should also raise and exception.
        """
        raw_story = ""
        story, summary = process_story(raw_story)
        self.assertEqual(story, [])
        self.assertEqual(summary, [])

    def test_story_with_missing_period(self):
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


if __name__ == "__main__":
    unittest.main()
