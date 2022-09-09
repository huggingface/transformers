# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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

from transformers.utils import flatten_dict


class GenericTester(unittest.TestCase):
    def test_flatten_dict(self):
        input_dict = {
            "task_specific_params": {
                "summarization": {"length_penalty": 1.0, "max_length": 128, "min_length": 12, "num_beams": 4},
                "summarization_cnn": {"length_penalty": 2.0, "max_length": 142, "min_length": 56, "num_beams": 4},
                "summarization_xsum": {"length_penalty": 1.0, "max_length": 62, "min_length": 11, "num_beams": 6},
            }
        }
        expected_dict = {
            "task_specific_params.summarization.length_penalty": 1.0,
            "task_specific_params.summarization.max_length": 128,
            "task_specific_params.summarization.min_length": 12,
            "task_specific_params.summarization.num_beams": 4,
            "task_specific_params.summarization_cnn.length_penalty": 2.0,
            "task_specific_params.summarization_cnn.max_length": 142,
            "task_specific_params.summarization_cnn.min_length": 56,
            "task_specific_params.summarization_cnn.num_beams": 4,
            "task_specific_params.summarization_xsum.length_penalty": 1.0,
            "task_specific_params.summarization_xsum.max_length": 62,
            "task_specific_params.summarization_xsum.min_length": 11,
            "task_specific_params.summarization_xsum.num_beams": 6,
        }

        self.assertEqual(flatten_dict(input_dict), expected_dict)
