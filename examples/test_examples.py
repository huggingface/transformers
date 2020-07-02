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


import argparse
import logging
import os
import sys
import unittest
from unittest.mock import patch


SRC_DIRS = [
    os.path.join(os.path.dirname(__file__), dirname)
    for dirname in ["text-generation", "text-classification", "language-modeling", "question-answering"]
]
sys.path.extend(SRC_DIRS)


if SRC_DIRS is not None:
    import run_generation
    import run_glue
    import run_language_modeling
    import run_squad


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def get_setup_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f")
    args = parser.parse_args()
    return args.f


class ExamplesTests(unittest.TestCase):
    def test_run_glue(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        testargs = """
            run_glue.py
            --model_name_or_path distilbert-base-uncased
            --data_dir ./tests/fixtures/tests_samples/MRPC/
            --task_name mrpc
            --do_train
            --do_eval
            --output_dir ./tests/fixtures/tests_samples/temp_dir
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --learning_rate=1e-4
            --max_steps=10
            --warmup_steps=2
            --overwrite_output_dir
            --seed=42
            --max_seq_length=128
            """.split()
        with patch.object(sys, "argv", testargs):
            result = run_glue.main()
            del result["eval_loss"]
            for value in result.values():
                self.assertGreaterEqual(value, 0.75)

    def test_run_language_modeling(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        # TODO: switch to smaller model like sshleifer/tiny-distilroberta-base

        testargs = """
            run_language_modeling.py
            --model_name_or_path distilroberta-base
            --model_type roberta
            --mlm
            --line_by_line
            --train_data_file ./tests/fixtures/sample_text.txt
            --eval_data_file ./tests/fixtures/sample_text.txt
            --output_dir ./tests/fixtures/tests_samples/temp_dir
            --overwrite_output_dir
            --do_train
            --do_eval
            --num_train_epochs=1
            --no_cuda
            """.split()
        with patch.object(sys, "argv", testargs):
            result = run_language_modeling.main()
            self.assertLess(result["perplexity"], 35)

    def test_run_squad(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        testargs = """
            run_squad.py
            --model_type=distilbert
            --model_name_or_path=sshleifer/tiny-distilbert-base-cased-distilled-squad
            --data_dir=./tests/fixtures/tests_samples/SQUAD
            --output_dir=./tests/fixtures/tests_samples/temp_dir
            --max_steps=10
            --warmup_steps=2
            --do_train
            --do_eval
            --version_2_with_negative
            --learning_rate=2e-4
            --per_gpu_train_batch_size=2
            --per_gpu_eval_batch_size=1
            --overwrite_output_dir
            --seed=42
        """.split()
        with patch.object(sys, "argv", testargs):
            result = run_squad.main()
            self.assertGreaterEqual(result["f1"], 25)
            self.assertGreaterEqual(result["exact"], 21)

    def test_generation(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        testargs = ["run_generation.py", "--prompt=Hello", "--length=10", "--seed=42"]
        model_type, model_name = ("--model_type=gpt2", "--model_name_or_path=sshleifer/tiny-gpt2")
        with patch.object(sys, "argv", testargs + [model_type, model_name]):
            result = run_generation.main()
            self.assertGreaterEqual(len(result[0]), 10)
