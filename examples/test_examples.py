# coding=utf-8
# Copyright 2018 HuggingFace Inc..
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest
import argparse
import logging

try:
    # python 3.4+ can use builtin unittest.mock instead of mock package
    from unittest.mock import patch
except ImportError:
    from mock import patch

import run_glue

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()

def get_setup_file():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    args = parser.parse_args()
    return args.f

class ExamplesTests(unittest.TestCase):

    def test_run_glue(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        testargs = ["run_glue.py", "--data_dir=./examples/tests_samples/MRPC/",
                    "--task_name=mrpc", "--do_train", "--do_eval", "--output_dir=./examples/tests_samples/temp_dir",
                    "--train_batch_size=4", "--eval_batch_size=2", "--num_train_epochs=2.0", "--overwrite_output_dir"]
        model_name = "--model_name=xlnet-large-cased"
        with patch.object(sys, 'argv', testargs + [model_name]):
            result = run_glue.main()
            for value in result.values():
                self.assertGreaterEqual(value, 0.75)


if __name__ == "__main__":
    unittest.main()
