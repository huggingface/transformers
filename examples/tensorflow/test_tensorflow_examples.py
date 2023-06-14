# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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
from unittest import skip
from unittest.mock import patch

import tensorflow as tf

from transformers.testing_utils import TestCasePlus, get_gpu_count, slow


SRC_DIRS = [
    os.path.join(os.path.dirname(__file__), dirname)
    for dirname in [
        "text-generation",
        "text-classification",
        "token-classification",
        "language-modeling",
        "multiple-choice",
        "question-answering",
        "summarization",
        "translation",
        "image-classification",
    ]
]
sys.path.extend(SRC_DIRS)


if SRC_DIRS is not None:
    import run_clm
    import run_image_classification
    import run_mlm
    import run_ner
    import run_qa as run_squad
    import run_summarization
    import run_swag
    import run_text_classification
    import run_translation


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


def is_cuda_available():
    return bool(tf.config.list_physical_devices("GPU"))


stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class ExamplesTests(TestCasePlus):
    @skip("Skipping until shape inference for to_tf_dataset PR is merged.")
    def test_run_text_classification(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_text_classification.py
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
            --max_steps=10
            --warmup_steps=2
            --seed=42
            --max_seq_length=128
            """.split()

        if is_cuda_available():
            testargs.append("--fp16")

        with patch.object(sys, "argv", testargs):
            run_text_classification.main()
            # Reset the mixed precision policy so we don't break other tests
            tf.keras.mixed_precision.set_global_policy("float32")
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_accuracy"], 0.75)

    def test_run_clm(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_clm.py
            --model_name_or_path distilgpt2
            --train_file ./tests/fixtures/sample_text.txt
            --validation_file ./tests/fixtures/sample_text.txt
            --do_train
            --do_eval
            --block_size 128
            --per_device_train_batch_size 2
            --per_device_eval_batch_size 1
            --num_train_epochs 2
            --output_dir {tmp_dir}
            --overwrite_output_dir
            """.split()

        if len(tf.config.list_physical_devices("GPU")) > 1:
            # Skipping because there are not enough batches to train the model + would need a drop_last to work.
            return

        with patch.object(sys, "argv", testargs):
            run_clm.main()
            result = get_results(tmp_dir)
            self.assertLess(result["eval_perplexity"], 100)

    def test_run_mlm(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_mlm.py
            --model_name_or_path distilroberta-base
            --train_file ./tests/fixtures/sample_text.txt
            --validation_file ./tests/fixtures/sample_text.txt
            --max_seq_length 64
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --do_train
            --do_eval
            --prediction_loss_only
            --num_train_epochs=1
            --learning_rate=1e-4
        """.split()

        with patch.object(sys, "argv", testargs):
            run_mlm.main()
            result = get_results(tmp_dir)
            self.assertLess(result["eval_perplexity"], 42)

    def test_run_ner(self):
        # with so little data distributed training needs more epochs to get the score on par with 0/1 gpu
        epochs = 7 if get_gpu_count() > 1 else 2

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_ner.py
            --model_name_or_path bert-base-uncased
            --train_file tests/fixtures/tests_samples/conll/sample.json
            --validation_file tests/fixtures/tests_samples/conll/sample.json
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --do_train
            --do_eval
            --warmup_steps=2
            --learning_rate=2e-4
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=2
            --num_train_epochs={epochs}
            --seed 7
        """.split()

        with patch.object(sys, "argv", testargs):
            run_ner.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["accuracy"], 0.75)

    def test_run_squad(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_qa.py
            --model_name_or_path bert-base-uncased
            --version_2_with_negative
            --train_file tests/fixtures/tests_samples/SQUAD/sample.json
            --validation_file tests/fixtures/tests_samples/SQUAD/sample.json
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --max_steps=10
            --warmup_steps=2
            --do_train
            --do_eval
            --learning_rate=2e-4
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
        """.split()

        with patch.object(sys, "argv", testargs):
            run_squad.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["f1"], 30)
            self.assertGreaterEqual(result["exact"], 30)

    def test_run_swag(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_swag.py
            --model_name_or_path bert-base-uncased
            --train_file tests/fixtures/tests_samples/swag/sample.json
            --validation_file tests/fixtures/tests_samples/swag/sample.json
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --max_steps=20
            --warmup_steps=2
            --do_train
            --do_eval
            --learning_rate=2e-4
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
        """.split()

        with patch.object(sys, "argv", testargs):
            run_swag.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["val_accuracy"], 0.8)

    @slow
    def test_run_summarization(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_summarization.py
            --model_name_or_path t5-small
            --train_file tests/fixtures/tests_samples/xsum/sample.json
            --validation_file tests/fixtures/tests_samples/xsum/sample.json
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --max_steps=50
            --warmup_steps=8
            --do_train
            --do_eval
            --learning_rate=2e-4
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
        """.split()

        with patch.object(sys, "argv", testargs):
            run_summarization.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["rouge1"], 10)
            self.assertGreaterEqual(result["rouge2"], 2)
            self.assertGreaterEqual(result["rougeL"], 7)
            self.assertGreaterEqual(result["rougeLsum"], 7)

    @slow
    def test_run_translation(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_translation.py
            --model_name_or_path Rocketknight1/student_marian_en_ro_6_1
            --source_lang en
            --target_lang ro
            --train_file tests/fixtures/tests_samples/wmt16/sample.json
            --validation_file tests/fixtures/tests_samples/wmt16/sample.json
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --warmup_steps=8
            --do_train
            --do_eval
            --learning_rate=3e-3
            --num_train_epochs 12
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --source_lang en_XX
            --target_lang ro_RO
        """.split()

        with patch.object(sys, "argv", testargs):
            run_translation.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["bleu"], 30)

    def test_run_image_classification(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_image_classification.py
            --dataset_name hf-internal-testing/cats_vs_dogs_sample
            --model_name_or_path microsoft/resnet-18
            --do_train
            --do_eval
            --learning_rate 1e-4
            --per_device_train_batch_size 2
            --per_device_eval_batch_size 1
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --dataloader_num_workers 16
            --num_train_epochs 2
            --train_val_split 0.1
            --seed 42
            --ignore_mismatched_sizes True
            """.split()

        with patch.object(sys, "argv", testargs):
            run_image_classification.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["accuracy"], 0.7)
