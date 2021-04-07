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

import dataclasses
import io
import json
import os
import sys
import unittest
from copy import deepcopy

from transformers import TrainingArguments
from transformers.file_utils import WEIGHTS_NAME
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
from test_trainer import TrainerIntegrationCommon, get_regression_trainer  # noqa


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
class TrainerIntegrationDeepSpeed(TestCasePlus, TrainerIntegrationCommon):
    """

    This class is for testing directly via get_regression_trainer

    It mixes in `TrainerIntegrationCommon` which already has a lot of helper validation methods which we can re-use here.
    """

    def setUp(self):
        super().setUp()

        args = TrainingArguments(".")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

        self.dist_env_1_gpu = dict(
            MASTER_ADDR="localhost", MASTER_PORT="10999", RANK="0", LOCAL_RANK="0", WORLD_SIZE="1"
        )
        self.ds_config_file = f"{self.test_file_dir_str}/ds_config.json"
        with io.open(self.ds_config_file, "r", encoding="utf-8") as f:
            self.ds_config_dict = json.load(f)

    def test_fake_notebook_no_launcher(self):
        # this setup emulates a notebook where a launcher needs to be emulated by hand
        with CaptureStd() as cs:  # noqa
            with mockenv_context(**self.dist_env_1_gpu):
                trainer = get_regression_trainer(local_rank=0, deepspeed=self.ds_config_file)
                trainer.train()
        # fixme:
        # assert "DeepSpeed info" in cs.out, "expected DeepSpeed logger output but got none"

    # Test various combos
    # 1. DS scheduler + DS optimizer: this is already tested by most other tests
    # 2. HF scheduler + HF optimizer:
    # 3. DS scheduler + HF optimizer:
    # 4. HF scheduler + DS optimizer:

    def test_hf_scheduler_hf_optimizer(self):
        a = 0
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_dict = deepcopy(self.ds_config_dict)
            del ds_config_dict["optimizer"]  # force default HF Trainer optimizer
            del ds_config_dict["scheduler"]  # force default HF Trainer scheduler
            ds_config_dict["zero_optimization"]["cpu_offload"] = False
            ds_config_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
            trainer = get_regression_trainer(a=a, local_rank=0, deepspeed=ds_config_dict)
            trainer.train()
        new_a = trainer.model.a.item()
        self.assertNotEqual(new_a, a)

    def test_ds_scheduler_hf_optimizer(self):
        a = 0
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_dict = deepcopy(self.ds_config_dict)
            del ds_config_dict["optimizer"]  # force default HF Trainer optimizer
            ds_config_dict["zero_optimization"]["cpu_offload"] = False
            ds_config_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
            trainer = get_regression_trainer(a=a, local_rank=0, deepspeed=ds_config_dict)
            trainer.train()
        new_a = trainer.model.a.item()
        self.assertNotEqual(new_a, a)

    def test_hf_scheduler_ds_optimizer(self):
        # this combo is not possible at the moment
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_dict = deepcopy(self.ds_config_dict)
            del ds_config_dict["scheduler"]  # force default HF Trainer scheduler
            ds_config_dict["zero_optimization"]["cpu_offload"] = False
            ds_config_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
            trainer = get_regression_trainer(local_rank=0, deepspeed=ds_config_dict)
            with self.assertRaises(Exception) as context:
                trainer.train()
        self.assertTrue("HF scheduler + DeepSpeed optimizer combination is not possible" in str(context.exception))

    def test_hf_optimizer_with_offload(self):
        # must not allow non-DS optimizer when using ZERO-offload
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_dict = deepcopy(self.ds_config_dict)
            del ds_config_dict["optimizer"]  # force default HF Trainer optimizer
            ds_config_dict["zero_optimization"]["cpu_offload"] = True
            # sanity check - should the default config change
            assert (
                "cpu_offload" in ds_config_dict["zero_optimization"]
                and ds_config_dict["zero_optimization"]["cpu_offload"] is True
            ), "ensure the config is set up correctly"
            trainer = get_regression_trainer(local_rank=0, deepspeed=ds_config_dict)
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

    def check_saved_checkpoints_deepspeed(self, output_dir, freq, total, is_pretrained=True):
        # adapted from TrainerIntegrationCommon.check_saved_checkpoints

        file_list = [WEIGHTS_NAME, "training_args.bin", "trainer_state.json", "config.json"]
        ds_file_list = ["mp_rank_00_model_states.pt", "zero_pp_rank_0_mp_rank_00optim_states.pt"]

        for step in range(freq, total, freq):
            checkpoint = os.path.join(output_dir, f"checkpoint-{step}")
            self.assertTrue(os.path.isdir(checkpoint))

            # common files
            for filename in file_list:
                self.assertTrue(os.path.isfile(os.path.join(checkpoint, filename)))

            # ds files
            ds_path = os.path.join(checkpoint, f"global_step{step}")
            for filename in ds_file_list:
                # filename = os.path.join(path, filename)
                # print(filename)
                self.assertTrue(os.path.isfile(os.path.join(ds_path, filename)))

    def test_save_checkpoints(self):
        # adapted from  TrainerIntegrationTest.test_save_checkpoints

        output_dir = self.get_auto_remove_tmp_dir()
        ds_config_dict = deepcopy(self.ds_config_dict)
        ds_config_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
        freq = 5

        # save checkpoints
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(
                output_dir=output_dir,
                save_steps=freq,
                deepspeed=ds_config_dict,
            )
            trainer.train()

        total = int(self.n_epochs * 64 / self.batch_size)
        self.check_saved_checkpoints_deepspeed(output_dir, freq, total)

    def test_can_resume_training(self):
        # adapted from TrainerIntegrationTest.test_can_resume_training

        output_dir = self.get_auto_remove_tmp_dir()
        ds_config_dict = deepcopy(self.ds_config_dict)
        ds_config_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
        kwargs = dict(output_dir=output_dir, train_len=128, save_steps=5, learning_rate=0.1, deepspeed=ds_config_dict)

        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(**kwargs)
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint = os.path.join(output_dir, "checkpoint-5")

            # Reinitialize trainer
            trainer = get_regression_trainer(**kwargs)

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

            # Now check with a later checkpoint that it also works when we span over one epoch
            checkpoint = os.path.join(output_dir, "checkpoint-15")

            # Reinitialize trainer and load model
            trainer = get_regression_trainer(**kwargs)

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

            # Now check failures

            # 1. fail to find a bogus checkpoint
            trainer = get_regression_trainer(**kwargs)
            with self.assertRaises(Exception) as context:
                trainer.train(resume_from_checkpoint=f"{checkpoint}-bogus")
            self.assertTrue("failed to resume from checkpoint" in str(context.exception))

            # 2. fail to find any checkpoint - due a fresh output_dir
            output_dir2 = self.get_auto_remove_tmp_dir()
            trainer = get_regression_trainer(output_dir=output_dir2, deepspeed=ds_config_dict)
            with self.assertRaises(Exception) as context:
                trainer.train(resume_from_checkpoint=True)
            self.assertTrue("No valid checkpoint found in output directory" in str(context.exception))


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
