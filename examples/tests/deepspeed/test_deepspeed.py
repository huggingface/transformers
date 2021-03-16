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

import io
import json
import os
import sys
import unittest
from copy import deepcopy

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
T5_SMALL = "t5-small"


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
class TestDeepSpeedWithoutLauncher(TestCasePlus):
    """

    This class is for testing directly via get_regression_trainer

    Note: if any of the tests of this class get run there will be at least one gpu occupied by them
    until this pytest worker exits. This is because the gpu memory allocated by the cuda-kernels
    won't be released until this pytest worker exits.

    This may appear as some run-away tests if you watch `nvidia-smi` while other tests that fork new
    processes are run. So there will be one or two "stale" processes reported in `nvidia-smi`. This
    is not a bug.

    """

    def setUp(self):
        super().setUp()
        self.dist_env_1_gpu = dict(
            MASTER_ADDR="localhost", MASTER_PORT="10999", RANK="0", LOCAL_RANK="0", WORLD_SIZE="1"
        )

        self.ds_config_zero2_file = f"{self.test_file_dir_str}/ds_config_zero2.json"
        self.ds_config_zero3_file = f"{self.test_file_dir_str}/ds_config_zero3.json"

        with io.open(self.ds_config_zero2_file, "r", encoding="utf-8") as f:
            self.ds_config_zero2_dict = json.load(f)
        with io.open(self.ds_config_zero3_file, "r", encoding="utf-8") as f:
            self.ds_config_zero3_dict = json.load(f)


    def test_fake_notebook_no_launcher(self):
        # this setup emulates a notebook where a launcher needs to be emulated by hand
        with CaptureStd() as cs:  # noqa
            with mockenv_context(**self.dist_env_1_gpu):
                trainer = get_regression_trainer(local_rank=0, deepspeed=self.ds_config_zero2_file)
                trainer.train()
        # XXX: the following check currently only works if run alone, see: https://github.com/microsoft/DeepSpeed/issues/810
        # assert "DeepSpeed info" in cs.out, "expected DeepSpeed logger output but got none"
        # so I'm not sure how to test that deepspeed actually did run, other than that it didn't fail

    # Test various combos
    # 1. DS scheduler + DS optimizer: this is already tested by most other tests
    # 2. HF scheduler + HF optimizer:
    # 3. DS scheduler + HF optimizer:
    # 4. HF scheduler + DS optimizer:

    def test_hf_scheduler_hf_optimizer(self):
        a = 0
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero2_dict = deepcopy(self.ds_config_zero2_dict)
            del ds_config_zero2_dict["optimizer"]  # force default HF Trainer optimizer
            del ds_config_zero2_dict["scheduler"]  # force default HF Trainer scheduler
            ds_config_zero2_dict["zero_optimization"]["cpu_offload"] = False
            ds_config_zero2_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
            trainer = get_regression_trainer(a=a, local_rank=0, deepspeed=ds_config_zero2_dict)
            trainer.train()
        new_a = trainer.model.a.item()
        self.assertNotEqual(new_a, a)

    def test_ds_scheduler_hf_optimizer(self):
        a = 0
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero2_dict = deepcopy(self.ds_config_zero2_dict)
            del ds_config_zero2_dict["optimizer"]  # force default HF Trainer optimizer
            ds_config_zero2_dict["zero_optimization"]["cpu_offload"] = False
            ds_config_zero2_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
            trainer = get_regression_trainer(a=a, local_rank=0, deepspeed=ds_config_zero2_dict)
            trainer.train()
        new_a = trainer.model.a.item()
        self.assertNotEqual(new_a, a)

    def test_hf_scheduler_ds_optimizer(self):
        # this combo is not possible at the moment
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero2_dict = deepcopy(self.ds_config_zero2_dict)
            del ds_config_zero2_dict["scheduler"]  # force default HF Trainer scheduler
            ds_config_zero2_dict["zero_optimization"]["cpu_offload"] = False
            ds_config_zero2_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
            trainer = get_regression_trainer(local_rank=0, deepspeed=ds_config_zero2_dict)
            with self.assertRaises(Exception) as context:
                trainer.train()
        self.assertTrue("HF scheduler + DeepSpeed optimizer combination is not possible" in str(context.exception))

    def test_hf_optimizer_with_offload(self):
        # must not allow non-DS optimizer when using ZERO-offload
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero2_dict = deepcopy(self.ds_config_zero2_dict)
            del ds_config_zero2_dict["optimizer"]  # force default HF Trainer optimizer
            ds_config_zero2_dict["zero_optimization"]["cpu_offload"] = True
            # sanity check - should the default config change
            assert (
                "cpu_offload" in ds_config_zero2_dict["zero_optimization"]
                and ds_config_zero2_dict["zero_optimization"]["cpu_offload"] is True
            ), "ensure the config is set up correctly"
            trainer = get_regression_trainer(local_rank=0, deepspeed=ds_config_zero2_dict)
            with self.assertRaises(Exception) as context:
                trainer.train()
        self.assertTrue("ZeRO Offload can only work with DeepSpeed optimizers" in str(context.exception))

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
                deepspeed=self.ds_config_zero2_file,
                per_device_train_batch_size=8,
                logging_steps=1,
            )
            trainer.train()
            post_train_a = trainer.model.a.item()

            # XXX: for some reason the following check fails with zero3 - not a broken but a
            # different qualitative outcome - need to investigate at some point

            # it's enough that train didn't fail for this test, but we must check that
            # optimizer/scheduler didn't run (since if it did this test isn't testing the right thing)
            self.assertEqual(post_train_a, a)

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
                deepspeed=self.ds_config_zero3_file,
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
                deepspeed=self.ds_config_zero3_file,
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
class TestDeepSpeedWithLauncher(TestCasePlus):
    """ This class is for testing via an external script """

    # Tests to devise #
    #
    # 1. predict_with_generate on multigpu - need to figure out how to give input sequences so that
    # the 2 gpus will generate prediction sequences that aren't of the same length - this is because
    # we had to code a special feature to sync the gpus when the predicted sequences aren't of the
    # same length. In general this will tested as a side-effect through a variety of other tests -
    # it'll simply hang trying to synchronize with other gpus if this problem is encountered. So as
    # long as we have a few full tests running on zero3 + predict_with_generate this should be
    # mostly covered.
    #
    # but there are 5 variations on beam search in `generate`- with identical code branched with `if
    # synced_gpus`

    @require_torch_multi_gpu
    def test_basic_distributed_zero2(self):
        self.run_quick(distributed=True, zero2=True)

    @require_torch_multi_gpu
    def test_basic_distributed_zero3(self):
        self.run_quick(distributed=True)

    def test_do_eval_no_train(self):
        # we should not fail if train is skipped
        self.run_quick(
            eval_steps=1,
            distributed=False,
            do_train=False,
            do_eval=True,
        )

    # XXX: need to do better validation beyond just that the run was successful
    def run_quick(
        self,
        eval_steps=10,
        distributed=True,
        zero2=False,
        do_train=True,
        do_eval=True,
        extra_args_str=None,
        remove_args_str=None,
    ):

        # we are doing quality testing so using a small real model
        output_dir = self.run_trainer(
            model_name=T5_SMALL,
            eval_steps=eval_steps,
            num_train_epochs=1,
            do_train=do_train,
            do_eval=do_eval,
            distributed=distributed,
            zero2=zero2,
            extra_args_str=extra_args_str,
            remove_args_str=remove_args_str,
        )

        if do_train:
            train_metrics = load_json(os.path.join(output_dir, "train_results.json"))
            self.assertIn("train_samples_per_second", train_metrics)
            self.assertGreater(train_metrics["train_samples_per_second"], 0.5)

        if do_eval:
            eval_metrics = load_json(os.path.join(output_dir, "eval_results.json"))
            self.assertIn("eval_bleu", eval_metrics)
            self.assertGreater(eval_metrics["eval_bleu"], 0)

    def run_trainer(
        self,
        model_name: str,
        eval_steps: int = 10,
        num_train_epochs: int = 1,
        do_train: bool = False,
        do_eval: bool = True,
        distributed: bool = True,
        zero2: bool = False,
        extra_args_str: str = None,
        remove_args_str: str = None,
    ):
        max_len = 32
        data_dir = self.examples_dir / "test_data/wmt_en_ro"
        # output_dir = self.get_auto_remove_tmp_dir()
        output_dir = "/tmp/zero3"
        args = f"""
            --model_name_or_path {model_name}
            --train_file {data_dir}/train.json
            --validation_file {data_dir}/val.json
            --output_dir {output_dir}
            --overwrite_output_dir
            --max_source_length {max_len}
            --max_target_length {max_len}
            --val_max_target_length {max_len}
            --warmup_steps 8
            --predict_with_generate
            --logging_steps 0
            --save_steps 0
            --eval_steps {eval_steps}
            --group_by_length
            --label_smoothing_factor 0.1
            --adafactor
            --source_lang en
            --target_lang ro
        """.split()
        args.extend(["--source_prefix", '"translate English to Romanian: "'])

        actions = 0
        if do_train:
            actions += 1
            args.extend(
                f"""
            --do_train
            --num_train_epochs {str(num_train_epochs)}
            --max_train_samples 100
            --per_device_train_batch_size 2
            --learning_rate 3e-3
            """.split()
            )

        if do_eval:
            actions += 1
            args.extend(
                f"""
            --do_eval
            --max_val_samples 100
            --per_device_eval_batch_size 2
            """.split()
            )

        assert actions > 0, "need at least do_train or do_eval for the test to run"

        if extra_args_str is not None:
            args.extend(extra_args_str.split())

        # currently only works for bool args
        if remove_args_str is not None:
            remove_args = remove_args_str.split()
            args = [x for x in args if x not in remove_args]

        ds_config = "ds_config_zero2.json" if zero2 else "ds_config_zero3.json"
        ds_args = f"--deepspeed {self.test_file_dir_str}/{ds_config}".split()
        script = [f"{self.examples_dir_str}/seq2seq/run_translation.py"]
        num_gpus = get_gpu_count() if distributed else 1
        launcher = f"deepspeed --num_gpus {num_gpus}".split()

        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        return output_dir
