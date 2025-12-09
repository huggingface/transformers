# Copyright 2025 HuggingFace Inc.
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

from transformers.conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightRenaming


class TestConversionMapping(unittest.TestCase):
    def test_register_checkpoint_conversion_mapping(self):
        register_checkpoint_conversion_mapping("foobar", [
            WeightRenaming(".block_sparse_moe.gate", ".mlp.gate"),
        ])
        self.assertEqual(len(get_checkpoint_conversion_mapping("foobar")), 1)

    def test_register_checkpoint_conversion_mapping_overwrites(self):
        register_checkpoint_conversion_mapping("foobarbaz", [
            WeightRenaming(".block_sparse_moe.gate", ".mlp.gate"),
        ])
        with self.assertRaises(ValueError):
            register_checkpoint_conversion_mapping("foobarbaz", [
                WeightRenaming(".block_sparse_moe.foo", ".mlp.foo"),
                WeightRenaming(".block_sparse_moe.bar", ".mlp.bar"),
            ])

        register_checkpoint_conversion_mapping("foobarbaz", [
            WeightRenaming(".block_sparse_moe.foo", ".mlp.foo"),
            WeightRenaming(".block_sparse_moe.bar", ".mlp.bar"),
        ], overwrite=True)

        self.assertEqual(len(get_checkpoint_conversion_mapping("foobarbaz")), 2)

