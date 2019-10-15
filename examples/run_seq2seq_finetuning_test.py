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

from .run_seq2seq_finetuning import process_story, _fit_to_block_size


class DataLoaderTest(unittest.TestCase):
    def __init__(self, block_size=10):
        self.block_size = block_size

    def source_and_target_too_small(self):
        """ When the sum of the lengths of the source and target sequences is
        smaller than the block size (minus the number of special tokens), skip the example. """
        src_seq = [1, 2, 3, 4]
        tgt_seq = [5, 6]
        self.assertEqual(_fit_to_block_size(src_seq, tgt_seq, self.block_size), None)

    def source_and_target_fit_exactly(self):
        """ When the sum of the lengths of the source and target sequences is
        equal to the block size (minus the number of special tokens), return the
        sequences unchanged. """
        src_seq = [1, 2, 3, 4]
        tgt_seq = [5, 6, 7]
        fitted_src, fitted_tgt = _fit_to_block_size(src_seq, tgt_seq, self.block_size)
        self.assertListEqual(src_seq == fitted_src)
        self.assertListEqual(tgt_seq == fitted_tgt)

    def source_too_big_target_ok(self):
        src_seq = [1, 2, 3, 4, 5, 6]
        tgt_seq = [1, 2]
        fitted_src, fitted_tgt = _fit_to_block_size(src_seq, tgt_seq, self.block_size)
        self.assertListEqual(src_seq == [1, 2, 3, 4, 5])
        self.assertListEqual(tgt_seq == fitted_tgt)

    def target_too_big_source_ok(self):
        src_seq = [1, 2, 3, 4]
        tgt_seq = [1, 2, 3, 4]
        fitted_src, fitted_tgt = _fit_to_block_size(src_seq, tgt_seq, self.block_size)
        self.assertListEqual(src_seq == src_seq)
        self.assertListEqual(tgt_seq == [1, 2, 3])

    def source_and_target_too_big(self):
        src_seq = [1, 2, 3, 4, 5, 6, 7]
        tgt_seq = [1, 2, 3, 4, 5, 6, 7]
        fitted_src, fitted_tgt = _fit_to_block_size(src_seq, tgt_seq, self.block_size)
        self.assertListEqual(src_seq == [1, 2, 3, 4, 5])
        self.assertListEqual(tgt_seq == [1, 2])


if __name__ == "__main__":
    unittest.main()
