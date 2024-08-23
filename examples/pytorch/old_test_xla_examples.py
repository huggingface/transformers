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


import json
import logging
import os
import sys
from time import time
from unittest.mock import patch

from transformers.testing_utils import TestCasePlus, require_torch_tpu


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def get_results(output_dir):
    results = {}
    path = os.path.join(output_dir, "all_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            results = json.load(f)
    else:
        raise ValueError(f"can't find {path}")
    return results


stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


@require_torch_tpu
class TorchXLAExamplesTests(TestCasePlus):
    def test_run_glue(self):
        import xla_spawn

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            ./examples/pytorch/text-classification/run_glue.py
            --num_cores=8
            ./examples/pytorch/text-classification/run_glue.py
            --model_name_or_path distilbert-base-uncased
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --train_file ./tests/fixtures/tests_samples/MRPC/train.csv
            --validation_file ./tests/fixtures/tests_samples/MRPC/dev.csv
            --do_train
            --do_eval
            --debug tpu_metrics_debug
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --learning_rate=1e-4
            --max_steps=10
            --warmup_steps=2
            --seed=42
            --max_seq_length=128
            """.split()

        with patch.object(sys, "argv", testargs):
            start = time()
            xla_spawn.main()
            end = time()

            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_accuracy"], 0.75)

            # Assert that the script takes less than 500 seconds to make sure it doesn't hang.
            self.assertLess(end - start, 500)

    def test_trainer_tpu(self):
        import xla_spawn

        testargs = """
            ./tests/test_trainer_tpu.py
            --num_cores=8
            ./tests/test_trainer_tpu.py
            """.split()
        with patch.object(sys, "argv", testargs):
            xla_spawn.main()
