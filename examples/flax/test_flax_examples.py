# coding=utf-8
# Copyright 2021 HuggingFace Inc.
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

from transformers.testing_utils import TestCasePlus, get_gpu_count, slow


SRC_DIRS = [
    os.path.join(os.path.dirname(__file__), dirname)
    for dirname in [
        "text-classification",
        "language-modeling",
        "summarization",
        "token-classification",
        "question-answering",
    ]
]
sys.path.extend(SRC_DIRS)


if SRC_DIRS is not None:
    import run_clm_flax
    import run_flax_glue
    import run_flax_ner
    import run_mlm_flax
    import run_qa
    import run_summarization_flax
    import run_t5_mlm_flax


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def get_setup_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f")
    args = parser.parse_args()
    return args.f


def get_results(output_dir, split="eval"):
    path = os.path.join(output_dir, f"{split}_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    raise ValueError(f"can't find {path}")


stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class ExamplesTests(TestCasePlus):
    def test_run_glue(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_glue.py
            --model_name_or_path distilbert-base-uncased
            --output_dir {tmp_dir}
            --train_file ./tests/fixtures/tests_samples/MRPC/train.csv
            --validation_file ./tests/fixtures/tests_samples/MRPC/dev.csv
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --learning_rate=1e-4
            --eval_steps=2
            --warmup_steps=2
            --seed=42
            --max_seq_length=128
            """.split()

        with patch.object(sys, "argv", testargs):
            run_flax_glue.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_accuracy"], 0.75)

    @slow
    def test_run_clm(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_clm_flax.py
            --model_name_or_path distilgpt2
            --train_file ./tests/fixtures/sample_text.txt
            --validation_file ./tests/fixtures/sample_text.txt
            --do_train
            --do_eval
            --block_size 128
            --per_device_train_batch_size 4
            --per_device_eval_batch_size 4
            --num_train_epochs 2
            --logging_steps 2 --eval_steps 2
            --output_dir {tmp_dir}
            --overwrite_output_dir
            """.split()

        with patch.object(sys, "argv", testargs):
            run_clm_flax.main()
            result = get_results(tmp_dir)
            self.assertLess(result["eval_perplexity"], 100)

    @slow
    def test_run_summarization(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_summarization.py
            --model_name_or_path t5-small
            --train_file tests/fixtures/tests_samples/xsum/sample.json
            --validation_file tests/fixtures/tests_samples/xsum/sample.json
            --test_file tests/fixtures/tests_samples/xsum/sample.json
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --num_train_epochs=3
            --warmup_steps=8
            --do_train
            --do_eval
            --do_predict
            --learning_rate=2e-4
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --predict_with_generate
        """.split()

        with patch.object(sys, "argv", testargs):
            run_summarization_flax.main()
            result = get_results(tmp_dir, split="test")
            self.assertGreaterEqual(result["test_rouge1"], 10)
            self.assertGreaterEqual(result["test_rouge2"], 2)
            self.assertGreaterEqual(result["test_rougeL"], 7)
            self.assertGreaterEqual(result["test_rougeLsum"], 7)

    @slow
    def test_run_mlm(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_mlm.py
            --model_name_or_path distilroberta-base
            --train_file ./tests/fixtures/sample_text.txt
            --validation_file ./tests/fixtures/sample_text.txt
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --max_seq_length 128
            --per_device_train_batch_size 4
            --per_device_eval_batch_size 4
            --logging_steps 2 --eval_steps 2
            --do_train
            --do_eval
            --num_train_epochs=1
        """.split()

        with patch.object(sys, "argv", testargs):
            run_mlm_flax.main()
            result = get_results(tmp_dir)
            self.assertLess(result["eval_perplexity"], 42)

    @slow
    def test_run_t5_mlm(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_t5_mlm_flax.py
            --model_name_or_path t5-small
            --train_file ./tests/fixtures/sample_text.txt
            --validation_file ./tests/fixtures/sample_text.txt
            --do_train
            --do_eval
            --max_seq_length 128
            --per_device_train_batch_size 4
            --per_device_eval_batch_size 4
            --num_train_epochs 2
            --logging_steps 2 --eval_steps 2
            --output_dir {tmp_dir}
            --overwrite_output_dir
            """.split()

        with patch.object(sys, "argv", testargs):
            run_t5_mlm_flax.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_accuracy"], 0.42)

    @slow
    def test_run_ner(self):
        # with so little data distributed training needs more epochs to get the score on par with 0/1 gpu
        epochs = 7 if get_gpu_count() > 1 else 2

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_flax_ner.py
            --model_name_or_path bert-base-uncased
            --train_file tests/fixtures/tests_samples/conll/sample.json
            --validation_file tests/fixtures/tests_samples/conll/sample.json
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --do_train
            --do_eval
            --warmup_steps=2
            --learning_rate=2e-4
            --logging_steps 2 --eval_steps 2
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=2
            --num_train_epochs={epochs}
            --seed 7
        """.split()

        with patch.object(sys, "argv", testargs):
            run_flax_ner.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_accuracy"], 0.75)
            self.assertGreaterEqual(result["eval_f1"], 0.3)

    @slow
    def test_run_qa(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_qa.py
            --model_name_or_path bert-base-uncased
            --version_2_with_negative
            --train_file tests/fixtures/tests_samples/SQUAD/sample.json
            --validation_file tests/fixtures/tests_samples/SQUAD/sample.json
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --num_train_epochs=3
            --warmup_steps=2
            --do_train
            --do_eval
            --logging_steps 2 --eval_steps 2
            --learning_rate=2e-4
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
        """.split()

        with patch.object(sys, "argv", testargs):
            run_qa.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_f1"], 30)
            self.assertGreaterEqual(result["eval_exact"], 30)
