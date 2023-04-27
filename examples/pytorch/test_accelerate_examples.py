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
import json
import logging
import os
import shutil
import sys
import tempfile
from unittest import mock

import torch
from accelerate.utils import write_basic_config

from transformers.testing_utils import TestCasePlus, get_gpu_count, run_command, slow, torch_device
from transformers.utils import is_apex_available


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


def is_cuda_and_apex_available():
    is_using_cuda = torch.cuda.is_available() and torch_device == "cuda"
    return is_using_cuda and is_apex_available()


stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class ExamplesTestsNoTrainer(TestCasePlus):
    @classmethod
    def setUpClass(cls):
        # Write Accelerate config, will pick up on CPU, GPU, and multi-GPU
        cls.tmpdir = tempfile.mkdtemp()
        cls.configPath = os.path.join(cls.tmpdir, "default_config.yml")
        write_basic_config(save_location=cls.configPath)
        cls._launch_args = ["accelerate", "launch", "--config_file", cls.configPath]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    @mock.patch.dict(os.environ, {"WANDB_MODE": "offline"})
    def test_run_glue_no_trainer(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            {self.examples_dir}/pytorch/text-classification/run_glue_no_trainer.py
            --model_name_or_path distilbert-base-uncased
            --output_dir {tmp_dir}
            --train_file ./tests/fixtures/tests_samples/MRPC/train.csv
            --validation_file ./tests/fixtures/tests_samples/MRPC/dev.csv
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --learning_rate=1e-4
            --seed=42
            --checkpointing_steps epoch
            --with_tracking
        """.split()

        if is_cuda_and_apex_available():
            testargs.append("--fp16")

        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        self.assertGreaterEqual(result["eval_accuracy"], 0.75)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "epoch_0")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "glue_no_trainer")))

    @mock.patch.dict(os.environ, {"WANDB_MODE": "offline"})
    def test_run_clm_no_trainer(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            {self.examples_dir}/pytorch/language-modeling/run_clm_no_trainer.py
            --model_name_or_path distilgpt2
            --train_file ./tests/fixtures/sample_text.txt
            --validation_file ./tests/fixtures/sample_text.txt
            --block_size 128
            --per_device_train_batch_size 5
            --per_device_eval_batch_size 5
            --num_train_epochs 2
            --output_dir {tmp_dir}
            --checkpointing_steps epoch
            --with_tracking
        """.split()

        if torch.cuda.device_count() > 1:
            # Skipping because there are not enough batches to train the model + would need a drop_last to work.
            return

        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        self.assertLess(result["perplexity"], 100)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "epoch_0")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "clm_no_trainer")))

    @mock.patch.dict(os.environ, {"WANDB_MODE": "offline"})
    def test_run_mlm_no_trainer(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            {self.examples_dir}/pytorch/language-modeling/run_mlm_no_trainer.py
            --model_name_or_path distilroberta-base
            --train_file ./tests/fixtures/sample_text.txt
            --validation_file ./tests/fixtures/sample_text.txt
            --output_dir {tmp_dir}
            --num_train_epochs=1
            --checkpointing_steps epoch
            --with_tracking
        """.split()

        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        self.assertLess(result["perplexity"], 42)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "epoch_0")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "mlm_no_trainer")))

    @mock.patch.dict(os.environ, {"WANDB_MODE": "offline"})
    def test_run_ner_no_trainer(self):
        # with so little data distributed training needs more epochs to get the score on par with 0/1 gpu
        epochs = 7 if get_gpu_count() > 1 else 2

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            {self.examples_dir}/pytorch/token-classification/run_ner_no_trainer.py
            --model_name_or_path bert-base-uncased
            --train_file tests/fixtures/tests_samples/conll/sample.json
            --validation_file tests/fixtures/tests_samples/conll/sample.json
            --output_dir {tmp_dir}
            --learning_rate=2e-4
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=2
            --num_train_epochs={epochs}
            --seed 7
            --checkpointing_steps epoch
            --with_tracking
        """.split()

        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        self.assertGreaterEqual(result["eval_accuracy"], 0.75)
        self.assertLess(result["train_loss"], 0.5)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "epoch_0")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "ner_no_trainer")))

    @mock.patch.dict(os.environ, {"WANDB_MODE": "offline"})
    def test_run_squad_no_trainer(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            {self.examples_dir}/pytorch/question-answering/run_qa_no_trainer.py
            --model_name_or_path bert-base-uncased
            --version_2_with_negative
            --train_file tests/fixtures/tests_samples/SQUAD/sample.json
            --validation_file tests/fixtures/tests_samples/SQUAD/sample.json
            --output_dir {tmp_dir}
            --seed=42
            --max_train_steps=10
            --num_warmup_steps=2
            --learning_rate=2e-4
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --checkpointing_steps epoch
            --with_tracking
        """.split()

        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        # Because we use --version_2_with_negative the testing script uses SQuAD v2 metrics.
        self.assertGreaterEqual(result["eval_f1"], 28)
        self.assertGreaterEqual(result["eval_exact"], 28)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "epoch_0")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "qa_no_trainer")))

    @mock.patch.dict(os.environ, {"WANDB_MODE": "offline"})
    def test_run_swag_no_trainer(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            {self.examples_dir}/pytorch/multiple-choice/run_swag_no_trainer.py
            --model_name_or_path bert-base-uncased
            --train_file tests/fixtures/tests_samples/swag/sample.json
            --validation_file tests/fixtures/tests_samples/swag/sample.json
            --output_dir {tmp_dir}
            --max_train_steps=20
            --num_warmup_steps=2
            --learning_rate=2e-4
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --with_tracking
        """.split()

        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        self.assertGreaterEqual(result["eval_accuracy"], 0.8)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "swag_no_trainer")))

    @slow
    @mock.patch.dict(os.environ, {"WANDB_MODE": "offline"})
    def test_run_summarization_no_trainer(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            {self.examples_dir}/pytorch/summarization/run_summarization_no_trainer.py
            --model_name_or_path t5-small
            --train_file tests/fixtures/tests_samples/xsum/sample.json
            --validation_file tests/fixtures/tests_samples/xsum/sample.json
            --output_dir {tmp_dir}
            --max_train_steps=50
            --num_warmup_steps=8
            --learning_rate=2e-4
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --checkpointing_steps epoch
            --with_tracking
        """.split()

        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        self.assertGreaterEqual(result["eval_rouge1"], 10)
        self.assertGreaterEqual(result["eval_rouge2"], 2)
        self.assertGreaterEqual(result["eval_rougeL"], 7)
        self.assertGreaterEqual(result["eval_rougeLsum"], 7)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "epoch_0")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "summarization_no_trainer")))

    @slow
    @mock.patch.dict(os.environ, {"WANDB_MODE": "offline"})
    def test_run_translation_no_trainer(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            {self.examples_dir}/pytorch/translation/run_translation_no_trainer.py
            --model_name_or_path sshleifer/student_marian_en_ro_6_1
            --source_lang en
            --target_lang ro
            --train_file tests/fixtures/tests_samples/wmt16/sample.json
            --validation_file tests/fixtures/tests_samples/wmt16/sample.json
            --output_dir {tmp_dir}
            --max_train_steps=50
            --num_warmup_steps=8
            --learning_rate=3e-3
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --source_lang en_XX
            --target_lang ro_RO
            --checkpointing_steps epoch
            --with_tracking
        """.split()

        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        self.assertGreaterEqual(result["eval_bleu"], 30)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "epoch_0")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "translation_no_trainer")))

    @slow
    def test_run_semantic_segmentation_no_trainer(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            {self.examples_dir}/pytorch/semantic-segmentation/run_semantic_segmentation_no_trainer.py
            --dataset_name huggingface/semantic-segmentation-test-sample
            --output_dir {tmp_dir}
            --max_train_steps=10
            --num_warmup_steps=2
            --learning_rate=2e-4
            --per_device_train_batch_size=2
            --per_device_eval_batch_size=1
            --checkpointing_steps epoch
        """.split()

        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        self.assertGreaterEqual(result["eval_overall_accuracy"], 0.10)

    @mock.patch.dict(os.environ, {"WANDB_MODE": "offline"})
    def test_run_image_classification_no_trainer(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            {self.examples_dir}/pytorch/image-classification/run_image_classification_no_trainer.py
            --model_name_or_path google/vit-base-patch16-224-in21k
            --dataset_name hf-internal-testing/cats_vs_dogs_sample
            --learning_rate 1e-4
            --per_device_train_batch_size 2
            --per_device_eval_batch_size 1
            --max_train_steps 2
            --train_val_split 0.1
            --seed 42
            --output_dir {tmp_dir}
            --with_tracking
            --checkpointing_steps 1
        """.split()

        if is_cuda_and_apex_available():
            testargs.append("--fp16")

        run_command(self._launch_args + testargs)
        result = get_results(tmp_dir)
        # The base model scores a 25%
        self.assertGreaterEqual(result["eval_accuracy"], 0.6)
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "step_1")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "image_classification_no_trainer")))
