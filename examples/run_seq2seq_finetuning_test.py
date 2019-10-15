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

from run_seq2seq_finetuning import _fit_to_block_size, process_story


class DataLoaderTest(unittest.TestCase):
    def setUp(self):
        self.block_size = 10

    def test_truncate_source_and_target_too_small(self):
        """ When the sum of the lengths of the source and target sequences is
        smaller than the block size (minus the number of special tokens), skip the example. """
        src_seq = [1, 2, 3, 4]
        tgt_seq = [5, 6]
        self.assertEqual(_fit_to_block_size(src_seq, tgt_seq, self.block_size), None)

    def test_truncate_source_and_target_fit_exactly(self):
        """ When the sum of the lengths of the source and target sequences is
        equal to the block size (minus the number of special tokens), return the
        sequences unchanged. """
        src_seq = [1, 2, 3, 4]
        tgt_seq = [5, 6, 7]
        fitted_src, fitted_tgt = _fit_to_block_size(src_seq, tgt_seq, self.block_size)
        self.assertListEqual(src_seq, fitted_src)
        self.assertListEqual(tgt_seq, fitted_tgt)

    def test_truncate_source_too_big_target_ok(self):
        src_seq = [1, 2, 3, 4, 5, 6]
        tgt_seq = [1, 2]
        fitted_src, fitted_tgt = _fit_to_block_size(src_seq, tgt_seq, self.block_size)
        self.assertListEqual(fitted_src, [1, 2, 3, 4, 5])
        self.assertListEqual(fitted_tgt, fitted_tgt)

    def test_truncate_target_too_big_source_ok(self):
        src_seq = [1, 2, 3, 4]
        tgt_seq = [1, 2, 3, 4]
        fitted_src, fitted_tgt = _fit_to_block_size(src_seq, tgt_seq, self.block_size)
        self.assertListEqual(fitted_src, src_seq)
        self.assertListEqual(fitted_tgt, [1, 2, 3])

    def test_truncate_source_and_target_too_big(self):
        src_seq = [1, 2, 3, 4, 5, 6, 7]
        tgt_seq = [1, 2, 3, 4, 5, 6, 7]
        fitted_src, fitted_tgt = _fit_to_block_size(src_seq, tgt_seq, self.block_size)
        self.assertListEqual(fitted_src, [1, 2, 3, 4, 5])
        self.assertListEqual(fitted_tgt, [1, 2])

    def test_process_story_no_highlights(self):
        """ Processing a story with no highlights should raise an exception.
        """
        raw_story = """It was the year of Our Lord one thousand seven hundred and
        seventy-five.\n\nSpiritual revelations were conceded to England at that
        favoured period, as at this."""
        with self.assertRaises(IndexError):
            process_story(raw_story)

    def test_process_empty_story(self):
        """ An empty story should also raise and exception.
        """
        raw_story = ""
        with self.assertRaises(IndexError):
            process_story(raw_story)

    def test_story_with_missing_period(self):
        raw_story = (
            "It was the year of Our Lord one thousand seven hundred and "
            "seventy-five\n\nSpiritual revelations were conceded to England "
            "at that favoured period, as at this.\n@highlight\n\nIt was the best of times"
        )
        story, summary = process_story(raw_story)

        expected_story = (
            "It was the year of Our Lord one thousand seven hundred and "
            "seventy-five. Spiritual revelations were conceded to England at that "
            "favoured period, as at this."
        )
        self.assertEqual(expected_story, story)

        expected_summary = "It was the best of times."
        self.assertEqual(expected_summary, summary)


if __name__ == "__main__":
    unittest.main()
