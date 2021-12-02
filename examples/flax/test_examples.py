# coding=utf-8
# Copyright 2021 HuggingFace Inc..
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


import argparse
import json
import logging
import os
import sys
from unittest.mock import patch

from transformers.testing_utils import CaptureLogger, TestCasePlus, get_gpu_count, slow, torch_device


SRC_DIRS = [
    os.path.join(os.path.dirname(__file__), dirname)
    for dirname in [
        "text-classification",
    ]
]
sys.path.extend(SRC_DIRS)


if SRC_DIRS is not None:
    import run_flax_glue


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def get_setup_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f")
    args = parser.parse_args()
    return args.f


def get_results(output_dir):
    results = {}
    path = os.path.join(output_dir, "all_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            results = json.load(f)
    else:
        raise ValueError(f"can't find {path}")
    return results


class ExamplesTests(TestCasePlus):
    def test_run_glue(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_glue.py
            --model_name_or_path distilbert-base-uncased
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --train_file ./tests/fixtures/tests_samples/MRPC/train.csv
            --validation_file ./tests/fixtures/tests_samples/MRPC/dev.csv
            --do_train
            --do_eval
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --learning_rate=1e-4
            --max_train_steps=10
            --num_warmup_steps=2
            --seed=42
            --max_length=128
            """.split()

        with patch.object(sys, "argv", testargs):
            run_flax_glue.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_accuracy"], 0.75)
