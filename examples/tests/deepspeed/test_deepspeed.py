# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import os
import sys
import unittest

from transformers.integrations import is_deepspeed_available
from transformers.testing_utils import (
    CaptureStd,
    TestCasePlus,
    execute_subprocess_async,
    get_gpu_count,
    mockenv_context,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
)
from transformers.trainer_utils import set_seed


bindir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(f"{bindir}/../../../tests")
from test_trainer import get_regression_trainer  # noqa


set_seed(42)
MBART_TINY = "sshleifer/tiny-mbart"


def load_json(path):
    with open(path) as f:
        return json.load(f)


# a candidate for testing_utils
def require_deepspeed(test_case):
    """
    Decorator marking a test that requires deepspeed
    """
    if not is_deepspeed_available():
        return unittest.skip("test requires deepspeed")(test_case)
    else:
        return test_case


@require_deepspeed
@require_torch_gpu
class TrainerIntegrationDeepSpeed(TestCasePlus):
    """ This class is for testing directly via get_regression_trainer """

    def setUp(self):
        super().setUp()
        self.dist_env_1_gpu = dict(
            MASTER_ADDR="localhost", MASTER_PORT="10999", RANK="0", LOCAL_RANK="0", WORLD_SIZE="1"
        )
        self.ds_config_file = f"{self.test_file_dir_str}/ds_config.json"

    def test_fake_notebook_no_launcher(self):

        # this setup emulates a notebook where a launcher needs to be emulated by hand

        with CaptureStd() as cs:
            with mockenv_context(**self.dist_env_1_gpu):
                trainer = get_regression_trainer(local_rank=0, deepspeed=self.ds_config_file)
                trainer.train()
        assert "DeepSpeed info" in cs.out, "expected DeepSpeed logger output but got none"

    def test_early_get_last_lr(self):
        # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
        # not run for the first few dozen steps while loss scale is too large, and thus during
        # that time `get_last_lr` will fail if called during that warm up stage,
        #
        # setting `logging_steps=1` forces an early `trainer._maybe_log_save_evaluate()` which calls
        # `self.lr_scheduler.get_last_lr()` and originally it'd fail on the very first step.
        with mockenv_context(**self.dist_env_1_gpu):
            a = b = 0.0
            trainer = get_regression_trainer(
                a=a,
                b=b,
                local_rank=0,
                train_len=8,
                deepspeed=self.ds_config_file,
                per_device_train_batch_size=8,
                logging_steps=1,
            )
            trainer.train()
            no_grad_accum_a = trainer.model.a.item()

            # it's enough that train didn't fail for this test, but we must check that
            # optimizer/scheduler didn't run (since if it did this test isn't testing the right thing)
            self.assertEqual(no_grad_accum_a, a)

    def test_gradient_accumulation(self):

        # this test measures that we get identical weights and similar loss with:
        # 1. per_device_train_batch_size=8, gradient_accumulation_steps=1
        # 2. per_device_train_batch_size=4, gradient_accumulation_steps=2
        # since the 2nd should produce the effective batch of 1st, with the same results
        #
        # I can get an identical loss for a small train_len=32, plus the power of the initial
        # dynamic loss scale value set to:
        #   "fp16.initial_scale_power": 1
        # plus having the same WarmupLR's warmup_min_lr == warmup_max_lr in the config file
        # but for some reason going to train_len=64 the weights, weights start to mismatch with this setup.
        # the culprit seems to be `initial_scale_power` - putting it back to its default 32 keeps the weights identical

        train_len = 64
        a = b = 0.0

        with mockenv_context(**self.dist_env_1_gpu):
            no_grad_accum_trainer = get_regression_trainer(
                a=a,
                b=b,
                local_rank=0,
                train_len=train_len,
                deepspeed=self.ds_config_file,
                per_device_train_batch_size=8,
                gradient_accumulation_steps=1,
            )
            no_grad_accum_result = no_grad_accum_trainer.train()
            no_grad_accum_loss = no_grad_accum_result.training_loss
            no_grad_accum_a = no_grad_accum_trainer.model.a.item()
            no_grad_accum_b = no_grad_accum_trainer.model.b.item()
            # make sure the optimizer kicked in - if it hasn't changed from the original value of a then make train_len bigger
            self.assertNotEqual(no_grad_accum_a, a)

        with mockenv_context(**self.dist_env_1_gpu):
            yes_grad_accum_trainer = get_regression_trainer(
                a=a,
                b=b,
                local_rank=0,
                train_len=train_len,
                deepspeed=self.ds_config_file,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2,
            )
            yes_grad_accum_result = yes_grad_accum_trainer.train()
            yes_grad_accum_loss = yes_grad_accum_result.training_loss
            yes_grad_accum_a = yes_grad_accum_trainer.model.a.item()
            yes_grad_accum_b = yes_grad_accum_trainer.model.b.item()
            self.assertNotEqual(yes_grad_accum_a, a)

        # training with half the batch size but accumulation steps as 2 should give the same weights
        self.assertEqual(no_grad_accum_a, yes_grad_accum_a)
        self.assertEqual(no_grad_accum_b, yes_grad_accum_b)

        # see the note above how to get identical loss on a small bs
        self.assertAlmostEqual(no_grad_accum_loss, yes_grad_accum_loss, places=5)


@slow
@require_deepspeed
@require_torch_gpu
class TestDeepSpeed(TestCasePlus):
    """ This class is for testing via an external script """

    @require_torch_multi_gpu
    def test_basic_distributed(self):
        self.run_quick(distributed=True)

    def test_do_eval_no_train(self):
        # we should not fail if train is skipped
        output_dir = self.run_trainer(
            eval_steps=1,
            max_len=12,
            model_name=MBART_TINY,
            num_train_epochs=1,
            distributed=False,
            extra_args_str="--do_eval",
            remove_args_str="--do_train",
        )
        val_metrics = load_json(os.path.join(output_dir, "eval_results.json"))
        assert "eval_bleu" in val_metrics

    # XXX: need to do better validation beyond just that the run was successful
    def run_quick(self, distributed=True, extra_args_str=None, remove_args_str=None):
        output_dir = self.run_trainer(
            eval_steps=1,
            max_len=12,
            model_name=MBART_TINY,
            num_train_epochs=1,
            distributed=distributed,
            extra_args_str=extra_args_str,
            remove_args_str=remove_args_str,
        )
        train_metrics = load_json(os.path.join(output_dir, "train_results.json"))
        assert "train_runtime" in train_metrics

    def run_trainer(
        self,
        eval_steps: int,
        max_len: str,
        model_name: str,
        num_train_epochs: int,
        distributed: bool = True,
        extra_args_str: str = None,
        remove_args_str: str = None,
    ):
        data_dir = self.examples_dir / "test_data/wmt_en_ro"
        output_dir = self.get_auto_remove_tmp_dir()
        args = f"""
            --model_name_or_path {model_name}
            --train_file {data_dir}/train.json
            --validation_file {data_dir}/val.json
            --output_dir {output_dir}
            --overwrite_output_dir
            --max_train_samples 8
            --max_val_samples 8
            --max_source_length {max_len}
            --max_target_length {max_len}
            --val_max_target_length {max_len}
            --do_train
            --num_train_epochs {str(num_train_epochs)}
            --per_device_train_batch_size 4
            --learning_rate 3e-3
            --warmup_steps 8
            --predict_with_generate
            --logging_steps 0
            --save_steps {str(eval_steps)}
            --group_by_length
            --label_smoothing_factor 0.1
            --adafactor
            --target_lang ro_RO
            --source_lang en_XX
        """.split()

        if extra_args_str is not None:
            args.extend(extra_args_str.split())

        if remove_args_str is not None:
            remove_args = remove_args_str.split()
            args = [x for x in args if x not in remove_args]

        ds_args = f"--deepspeed {self.test_file_dir_str}/ds_config.json".split()
        script = [f"{self.examples_dir_str}/seq2seq/run_translation.py"]
        num_gpus = get_gpu_count() if distributed else 1
        launcher = f"deepspeed --num_gpus {num_gpus}".split()

        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"PYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        return output_dir
