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
import unittest
from copy import deepcopy

from parameterized import parameterized
from transformers import AutoModel, TrainingArguments, is_torch_available, logging
from transformers.deepspeed import HfDeepSpeedConfig, is_deepspeed_available
from transformers.file_utils import WEIGHTS_NAME
from transformers.testing_utils import (
    CaptureLogger,
    CaptureStd,
    CaptureStderr,
    ExtendSysPath,
    LoggingLevel,
    TestCasePlus,
    execute_subprocess_async,
    get_gpu_count,
    mockenv_context,
    require_deepspeed,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
)
from transformers.trainer_utils import get_last_checkpoint, set_seed


tests_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
root_dir = os.path.dirname(tests_dir)
with ExtendSysPath(tests_dir):
    from test_trainer import TrainerIntegrationCommon  # noqa

    if is_torch_available():
        from test_trainer import RegressionModelConfig, RegressionPreTrainedModel, get_regression_trainer  # noqa


set_seed(42)

# default torch.distributed port
DEFAULT_MASTER_PORT = "10999"

T5_SMALL = "t5-small"
T5_TINY = "patrickvonplaten/t5-tiny-random"
GPT2_TINY = "sshleifer/tiny-gpt2"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_master_port(real_launcher=False):
    """
    When using a single gpu launcher emulation (i.e. not deepspeed or python -m torch.distributed)
    the issue is that once the port is tied it can't be used anywhere else outside of this process,
    since torch.dist doesn't free the port until the process exits. Therefore for the sake of being
    able to run both emulated launcher and normal launcher tests we need 2 distinct ports.

    This function will give the right port in the right context. For real launcher it'll give the
    base port, for emulated launcher it'll give the base port + 1. In both cases a string is
    returned.

    Args:
        `real_launcher`: whether a real launcher is going to be used, or the emulated one

    """

    master_port_base = os.environ.get("DS_TEST_PORT", DEFAULT_MASTER_PORT)
    if not real_launcher:
        master_port_base = str(int(master_port_base) + 1)
    return master_port_base


def require_deepspeed_aio(test_case):
    """
    Decorator marking a test that requires deepspeed aio (nvme)
    """
    if not is_deepspeed_available():
        return unittest.skip("test requires deepspeed")(test_case)

    import deepspeed
    from deepspeed.ops.aio import AsyncIOBuilder

    if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
        return unittest.skip("test requires deepspeed async-io")(test_case)
    else:
        return test_case


if is_deepspeed_available():
    from deepspeed.utils import logger as deepspeed_logger  # noqa
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
    from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled  # noqa


def get_launcher(distributed=False):
    # 1. explicitly set --num_nodes=1 just in case these tests end up run on a multi-node setup
    # - it won't be able to handle that
    # 2. for now testing with just 2 gpus max (since some quality tests may give different
    # results with mode gpus because we use very little data)
    num_gpus = min(2, get_gpu_count()) if distributed else 1
    master_port = get_master_port(real_launcher=True)
    return f"deepspeed --num_nodes 1 --num_gpus {num_gpus} --master_port {master_port}".split()


ZERO2 = "zero2"
ZERO3 = "zero3"
stages = [ZERO2, ZERO3]


@require_deepspeed
@require_torch_gpu
class CoreIntegrationDeepSpeed(TestCasePlus, TrainerIntegrationCommon):
    """
    Testing non-Trainer DeepSpeed integration
    """

    def setUp(self):
        super().setUp()

        master_port = get_master_port(real_launcher=False)
        self.dist_env_1_gpu = dict(
            MASTER_ADDR="localhost", MASTER_PORT=master_port, RANK="0", LOCAL_RANK="0", WORLD_SIZE="1"
        )

    def test_init_zero3(self):
        # test that zero.Init() works correctly under zero3
        ds_config = {
            "train_batch_size": 1,
            "zero_optimization": {
                "stage": 3,
            },
        }

        dschf = HfDeepSpeedConfig(ds_config)

        self.assertTrue(dschf.is_zero3())
        self.assertTrue(is_deepspeed_zero3_enabled())

        with LoggingLevel(logging.INFO):
            with mockenv_context(**self.dist_env_1_gpu):
                logger = logging.get_logger("transformers.modeling_utils")
                with CaptureLogger(logger) as cl:
                    AutoModel.from_pretrained(T5_TINY)
        self.assertIn("Detected DeepSpeed ZeRO-3", cl.out)

        # now remove zero optimization
        del ds_config["zero_optimization"]
        dschf = HfDeepSpeedConfig(ds_config)

        self.assertFalse(dschf.is_zero3())
        self.assertFalse(is_deepspeed_zero3_enabled())

        with LoggingLevel(logging.INFO):
            with mockenv_context(**self.dist_env_1_gpu):
                logger = logging.get_logger("transformers.modeling_utils")
                with CaptureLogger(logger) as cl:
                    AutoModel.from_pretrained(T5_TINY)
        self.assertNotIn("Detected DeepSpeed ZeRO-3", cl.out)


@require_deepspeed
@require_torch_gpu
class TrainerIntegrationDeepSpeed(TestCasePlus, TrainerIntegrationCommon):
    """

    This class is for testing directly via get_regression_trainer

    It mixes in `TrainerIntegrationCommon` which already has a lot of helper validation methods
    which we can re-use here.

    Important: this class' setup can only work with a single gpu because it runs within the current
    pytest worker. For multi-gpu tests use TestDeepSpeedWithLauncher.

    Note: if any of the tests of this class get run there will be at least one gpu occupied by them
    until this pytest worker exits. This is because the gpu memory allocated by the cuda-kernels
    won't be released until this pytest worker exits.

    This may appear as some run-away tests if you watch `nvidia-smi` while other tests that fork new
    processes are run. So there will be one or two "stale" processes reported in `nvidia-smi`. This
    is not a bug.
    """

    def setUp(self):
        super().setUp()

        args = TrainingArguments(".")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

        master_port = get_master_port(real_launcher=False)
        self.dist_env_1_gpu = dict(
            MASTER_ADDR="localhost", MASTER_PORT=master_port, RANK="0", LOCAL_RANK="0", WORLD_SIZE="1"
        )

        self.ds_config_file = dict(
            zero2=f"{self.test_file_dir_str}/ds_config_zero2.json",
            zero3=f"{self.test_file_dir_str}/ds_config_zero3.json",
        )

        # use self.get_config_dict(stage) to use these to ensure the original is not modified
        with io.open(self.ds_config_file[ZERO2], "r", encoding="utf-8") as f:
            config_zero2 = json.load(f)
            # by default use fp16
            config_zero2["fp16"]["enabled"] = True
        with io.open(self.ds_config_file[ZERO3], "r", encoding="utf-8") as f:
            config_zero3 = json.load(f)
            # by default use fp16
            config_zero3["fp16"]["enabled"] = True
            # This setting slows things down, so don't enable it by default unless needed by a test.
            # It's in the file as a demo for users since we want everything to work out of the box even if slower.
            config_zero3["zero_optimization"]["stage3_gather_fp16_weights_on_model_save"] = False
        self.ds_config_dict = dict(
            zero2=config_zero2,
            zero3=config_zero3,
        )

    def get_config_dict(self, stage):
        # As some tests modify the dict, always make a copy
        return deepcopy(self.ds_config_dict[stage])

    # --- These tests are enough to run on one of zero stages --- #

    def test_hf_ds_config_mismatch(self):

        ds_config = self.get_config_dict(ZERO2)

        # Purposefully configure these values to mismatch TrainingArguments values.
        # This currently doesn't cover all keys (but it could)
        per_device_train_batch_size = 2
        ds_config["train_micro_batch_size_per_gpu"] = per_device_train_batch_size + 2

        ds_config["train_batch_size"] = 1000

        gradient_accumulation_steps = 2
        ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps + 2

        max_grad_norm = 1.0
        ds_config["gradient_clipping"] = max_grad_norm + 0.1

        adam_beta1, adam_beta2 = 0.9, 0.99
        ds_config["optimizer"]["params"]["betas"] = [adam_beta1 - 0.1, adam_beta2 - 0.1]

        fp16 = True
        ds_config["fp16"]["enabled"] = not fp16

        keys = [
            "per_device_train_batch_size",
            "train_batch_size",
            "gradient_accumulation_steps",
            "max_grad_norm",
            "betas",
            "fp16",
        ]

        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(
                local_rank=0,
                fp16=fp16,
                deepspeed=ds_config,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                adam_beta1=adam_beta1,
                adam_beta2=adam_beta2,
            )
            with self.assertRaises(Exception) as context:
                trainer.train()

        for key in keys:
            self.assertTrue(
                key in str(context.exception),
                f"{key} is not in the exception message:\n{context.exception}",
            )

    # Test various combos
    # 1. DS scheduler + DS optimizer: this is already tested by most other tests
    # 2. HF scheduler + HF optimizer:
    # 3. DS scheduler + HF optimizer:
    # 4. HF scheduler + DS optimizer:

    def test_hf_scheduler_hf_optimizer(self):
        a = 0
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero2_dict = self.get_config_dict(ZERO2)
            del ds_config_zero2_dict["optimizer"]  # force default HF Trainer optimizer
            del ds_config_zero2_dict["scheduler"]  # force default HF Trainer scheduler
            ds_config_zero2_dict["zero_optimization"]["offload_optimizer"]["device"] = "none"
            ds_config_zero2_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
            trainer = get_regression_trainer(a=a, local_rank=0, fp16=True, deepspeed=ds_config_zero2_dict)
            trainer.train()
        new_a = trainer.model.a.item()
        self.assertNotEqual(new_a, a)

    def test_ds_scheduler_hf_optimizer(self):
        a = 0
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero2_dict = self.get_config_dict(ZERO2)
            del ds_config_zero2_dict["optimizer"]  # force default HF Trainer optimizer
            ds_config_zero2_dict["zero_optimization"]["offload_optimizer"]["device"] = "none"
            ds_config_zero2_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
            trainer = get_regression_trainer(a=a, local_rank=0, fp16=True, deepspeed=ds_config_zero2_dict)
            trainer.train()
        new_a = trainer.model.a.item()
        self.assertNotEqual(new_a, a)

    def test_hf_scheduler_ds_optimizer(self):
        a = 0
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero2_dict = self.get_config_dict(ZERO2)
            del ds_config_zero2_dict["scheduler"]  # force default HF Trainer scheduler
            ds_config_zero2_dict["zero_optimization"]["offload_optimizer"]["device"] = "none"
            ds_config_zero2_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
            trainer = get_regression_trainer(local_rank=0, fp16=True, deepspeed=ds_config_zero2_dict)
            trainer.train()
        new_a = trainer.model.a.item()
        self.assertNotEqual(new_a, a)

    @require_deepspeed_aio
    def test_stage3_nvme_offload(self):
        with mockenv_context(**self.dist_env_1_gpu):
            # this actually doesn't have to be on NVMe, any storage will do since this test only
            # runs a simple check that we can use some directory as if it were NVMe
            nvme_path = self.get_auto_remove_tmp_dir()
            nvme_config = dict(device="nvme", nvme_path=nvme_path)
            ds_config_zero3_dict = self.get_config_dict(ZERO3)
            ds_config_zero3_dict["zero_optimization"]["offload_optimizer"] = nvme_config
            ds_config_zero3_dict["zero_optimization"]["offload_param"] = nvme_config
            trainer = get_regression_trainer(local_rank=0, fp16=True, deepspeed=ds_config_zero3_dict)
            with CaptureLogger(deepspeed_logger) as cl:
                trainer.train()
            self.assertIn("DeepSpeed info", cl.out, "expected DeepSpeed logger output but got none")

    # --- These tests need to run on both zero stages --- #

    @parameterized.expand(stages)
    def test_hf_optimizer_with_offload(self, stage):
        # non-DS optimizers can be used with ZERO-offload (as long as they have both CPU and GPU implementation (except LAMB))
        ds_config_dict = self.get_config_dict(stage)
        del ds_config_dict["optimizer"]  # force default HF Trainer optimizer
        # force cpu offload
        ds_config_dict["zero_optimization"]["offload_optimizer"]["device"] = "cpu"
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(local_rank=0, fp16=True, deepspeed=ds_config_dict)
            with CaptureLogger(deepspeed_logger) as cl:
                trainer.train()
            self.assertIn("DeepSpeed info", cl.out, "expected DeepSpeed logger output but got none")

    @parameterized.expand(stages)
    def test_fake_notebook_no_launcher(self, stage):
        # this setup emulates a notebook where a launcher needs to be emulated by hand

        # note that unittest resets sys.stdout each test, so `CaptureStd` will work here to capture
        # DeepSpeed log if this test happens to run first in this pytest worker. But it will fail if
        # it's run not as a first test as `sys.stdout` will no longer be the same. So we either have
        # to reset `deepspeed_logger.handlers[0].setStream(sys.stdout)` or directly capture from the deepspeed_logger.
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(local_rank=0, fp16=True, deepspeed=self.get_config_dict(stage))
            with CaptureLogger(deepspeed_logger) as cl:
                trainer.train()
            self.assertIn("DeepSpeed info", cl.out, "expected DeepSpeed logger output but got none")

    @parameterized.expand(stages)
    def test_early_get_last_lr(self, stage):
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
                fp16=True,
                deepspeed=self.get_config_dict(stage),
                per_device_train_batch_size=8,
                logging_steps=1,
            )
            trainer.train()
            post_train_a = trainer.model.a.item()

            # XXX: for some reason the following check fails with zero3 - not a broken but a
            # different qualitative outcome - as if optimizer did run
            # oddly getting 1.0 for both a and b from 0.0 - there is a bug somewhere
            # print(trainer.model.a.item())
            # print(trainer.model.b.item())
            # need to investigate at some point
            if stage == ZERO3:
                return

            # it's enough that train didn't fail for this test, but we must check that
            # optimizer/scheduler didn't run (since if it did this test isn't testing the right thing)
            self.assertEqual(post_train_a, a)

    @parameterized.expand(stages)
    def test_gradient_accumulation(self, stage):
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

        kwargs = dict(
            a=a,
            b=b,
            local_rank=0,
            train_len=train_len,
            fp16=True,
            deepspeed=self.get_config_dict(stage),
        )

        with mockenv_context(**self.dist_env_1_gpu):
            no_grad_accum_trainer = get_regression_trainer(
                **kwargs,
                per_device_train_batch_size=16,
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
                **kwargs,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
            )
            yes_grad_accum_result = yes_grad_accum_trainer.train()
            yes_grad_accum_loss = yes_grad_accum_result.training_loss
            yes_grad_accum_a = yes_grad_accum_trainer.model.a.item()
            yes_grad_accum_b = yes_grad_accum_trainer.model.b.item()
            self.assertNotEqual(yes_grad_accum_a, a)

        # training with half the batch size but accumulation steps as 2 should give the same
        # weights, but sometimes get a slight difference still of 1e-6
        self.assertAlmostEqual(no_grad_accum_a, yes_grad_accum_a, places=5)
        self.assertAlmostEqual(no_grad_accum_b, yes_grad_accum_b, places=5)

        # see the note above how to get identical loss on a small bs
        self.assertAlmostEqual(no_grad_accum_loss, yes_grad_accum_loss, places=2)

    def check_saved_checkpoints_deepspeed(self, output_dir, freq, total, stage):
        # adapted from TrainerIntegrationCommon.check_saved_checkpoints

        file_list = [WEIGHTS_NAME, "training_args.bin", "trainer_state.json", "config.json"]

        if stage == ZERO2:
            ds_file_list = ["mp_rank_00_model_states.pt"]
        elif stage == ZERO3:
            ds_file_list = ["zero_pp_rank_0_mp_rank_00_model_states.pt"]
        else:
            raise ValueError(f"unknown stage {stage}")

        # XXX: this can be recoded and then removed once we require deepspeed>0.3.13
        from packaging import version

        import deepspeed

        if version.parse(deepspeed.__version__) > version.parse("0.3.13"):
            ds_file_list.append("zero_pp_rank_0_mp_rank_00_optim_states.pt")
        else:
            ds_file_list.append("zero_pp_rank_0_mp_rank_00optim_states.pt")

        for step in range(freq, total, freq):
            checkpoint = os.path.join(output_dir, f"checkpoint-{step}")
            self.assertTrue(os.path.isdir(checkpoint), f"[{stage}] {checkpoint} dir is not found")

            # common files
            for filename in file_list:
                path = os.path.join(checkpoint, filename)
                self.assertTrue(os.path.isfile(path), f"[{stage}] {path} is not found")

            # ds files
            ds_path = os.path.join(checkpoint, f"global_step{step}")
            for filename in ds_file_list:
                # filename = os.path.join(path, filename)
                # print(filename)
                path = os.path.join(ds_path, filename)
                self.assertTrue(os.path.isfile(path), f"[{stage}] {path} is not found")

    @parameterized.expand(stages)
    def test_save_checkpoints(self, stage):
        # adapted from  TrainerIntegrationTest.test_save_checkpoints

        freq = 5
        output_dir = self.get_auto_remove_tmp_dir()
        ds_config_dict = self.get_config_dict(stage)
        ds_config_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
        if stage == ZERO3:
            ds_config_dict["zero_optimization"]["stage3_gather_fp16_weights_on_model_save"] = True

        # save checkpoints
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(
                output_dir=output_dir,
                save_steps=freq,
                fp16=True,
                deepspeed=ds_config_dict,
            )
            trainer.train()

        total = int(self.n_epochs * 64 / self.batch_size)
        self.check_saved_checkpoints_deepspeed(output_dir, freq, total, stage)

    @parameterized.expand(stages)
    def test_can_resume_training_errors(self, stage):

        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_dict = self.get_config_dict(stage)
            output_dir = self.get_auto_remove_tmp_dir()
            trainer = get_regression_trainer(output_dir=output_dir, fp16=True, deepspeed=ds_config_dict)

            # 1. fail to find any checkpoint - due a fresh output_dir
            with self.assertRaises(Exception) as context:
                trainer.train(resume_from_checkpoint=True)
            self.assertTrue(
                "No valid checkpoint found in output directory" in str(context.exception),
                f"got exception: {context.exception}",
            )

            # 2. fail to find a bogus checkpoint
            with self.assertRaises(Exception) as context:
                checkpoint = os.path.join(output_dir, "checkpoint-5")
                trainer.train(resume_from_checkpoint=f"{checkpoint}-bogus")
            self.assertTrue(
                "Can't find a valid checkpoint at" in str(context.exception), f"got exception: {context.exception}"
            )

    @parameterized.expand(stages)
    def test_can_resume_training_normal(self, stage):
        # adapted from TrainerIntegrationTest.test_can_resume_training
        # test normal resume for each stage separately, error-handling is tested in a different test
        output_dir = self.get_auto_remove_tmp_dir()
        ds_config_dict = self.get_config_dict(stage)
        ds_config_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
        if stage == ZERO3:
            ds_config_dict["zero_optimization"]["stage3_gather_fp16_weights_on_model_save"] = True

        kwargs = dict(
            output_dir=output_dir, train_len=128, save_steps=5, learning_rate=0.1, fp16=True, deepspeed=ds_config_dict
        )

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

            # Finally, should be able to resume with the same trainer/same deepspeed engine instance
            # XXX: but currently this not possible due DS bug: https://github.com/microsoft/DeepSpeed/issues/1612
            # trainer.train(resume_from_checkpoint=checkpoint)
            # a workaround needs to be used that re-creates the deepspeed engine

    @parameterized.expand(stages)
    def test_load_state_dict_from_zero_checkpoint(self, stage):
        # test that we can load fp32 weights directly from the zero checkpoint into the current model

        output_dir = self.get_auto_remove_tmp_dir()  # "./xxx", after=False, before=False)

        ds_config_dict = self.get_config_dict(stage)

        kwargs = dict(
            output_dir=output_dir,
            train_len=4,
            per_device_train_batch_size=4,
            num_train_epochs=1,
            save_strategy="steps",
            save_steps=1,
            learning_rate=0.1,
            fp16=True,
            deepspeed=ds_config_dict,
        )

        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(**kwargs)
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint_dir = get_last_checkpoint(output_dir)
            model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)

            (a1, b1) = model.a.item(), model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

    def test_config_object(self):
        # test that we can switch from zero2 to zero3 in the same process for example
        # test is_zero, etc.
        output_dir = self.get_auto_remove_tmp_dir()
        kwargs = dict(output_dir=output_dir, train_len=8, fp16=True)

        ds_config_zero3_dict = self.get_config_dict("zero3")
        ds_config_zero2_dict = self.get_config_dict("zero2")

        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(deepspeed=ds_config_zero3_dict, **kwargs)
            self.assertTrue(is_deepspeed_zero3_enabled())

            # test we can repeat that and with train this time
            trainer = get_regression_trainer(deepspeed=ds_config_zero3_dict, **kwargs)
            trainer.train()
            self.assertTrue(is_deepspeed_zero3_enabled())

            # test zero3 is disabled
            trainer = get_regression_trainer(deepspeed=ds_config_zero2_dict, **kwargs)
            self.assertFalse(is_deepspeed_zero3_enabled())

            # check config obj
            config = deepspeed_config()
            self.assertTrue(bool(config), "Deepspeed config should be accessible")

            del trainer
            # now weakref should gc the global and we shouldn't get anything here
            config = deepspeed_config()
            self.assertFalse(is_deepspeed_zero3_enabled())
            self.assertFalse(bool(config), "Deepspeed config should not be accessible")


@slow
@require_deepspeed
@require_torch_gpu
class TestDeepSpeedWithLauncher(TestCasePlus):
    """This class is for testing via an external script - can do multiple gpus"""

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
    #
    # 2. most tests should probably be run on both: zero2 and zero3 configs
    #

    @require_torch_multi_gpu
    @parameterized.expand(stages)
    def test_basic_distributed(self, stage):
        self.run_and_check(stage=stage, distributed=True)

    def test_do_eval_no_train(self):
        # testing only zero3 since zero2 makes no sense with inference
        self.run_and_check(
            stage=ZERO3,
            eval_steps=1,
            distributed=False,
            do_train=False,
            do_eval=True,
        )

    @parameterized.expand(stages)
    def test_fp32_non_distributed(self, stage):
        # real model needs too much GPU memory under stage2+fp32, so using tiny random model here -
        # therefore no quality checks, just basic completion checks are done
        self.run_and_check(
            stage=stage,
            model_name=T5_TINY,
            distributed=False,
            do_train=True,
            do_eval=True,
            quality_checks=False,
            fp16=False,
        )

    @require_torch_multi_gpu
    @parameterized.expand(stages)
    def test_fp32_distributed(self, stage):
        # real model needs too much GPU memory under stage2+fp32, so using tiny random model here -
        # therefore no quality checks, just basic completion checks are done
        self.run_and_check(
            stage=stage,
            model_name=T5_TINY,
            distributed=True,
            do_train=True,
            do_eval=True,
            quality_checks=False,
            fp16=False,
        )

    @parameterized.expand(stages)
    def test_resume_train_not_from_ds_checkpoint(self, stage):
        # do normal training and then resume not from the deepspeed checkpoint but explicitly from
        # the saved model dir

        do_train = True
        do_eval = False
        kwargs = dict(stage=stage, eval_steps=1, distributed=True, do_train=do_train, do_eval=do_eval)

        # 1. normal training
        output_dir = self.run_and_check(**kwargs)

        # 2. now resume explicitly from the saved weights, by passing --model_name_or_path output_dir
        # - i.e. the same path the model was saved to in step 1
        output_dir = self.run_trainer(**kwargs, model_name=output_dir)

        self.do_checks(output_dir, do_train=do_train, do_eval=do_eval)

    @require_torch_multi_gpu
    @parameterized.expand(["fp16", "fp32"])
    def test_inference(self, dtype):
        # this is just inference, so no optimizer should be loaded
        # it only works for z3 (makes no sense with z1-z2)
        fp16 = True if dtype == "fp16" else False
        self.run_and_check(
            stage=ZERO3,
            model_name=T5_TINY,
            distributed=True,
            do_train=False,
            do_eval=True,
            quality_checks=False,
            fp16=fp16,
        )

    def do_checks(self, output_dir, do_train=True, do_eval=True, quality_checks=True):

        if do_train:
            train_metrics = load_json(os.path.join(output_dir, "train_results.json"))
            self.assertIn("train_samples_per_second", train_metrics)
            if quality_checks:
                self.assertGreater(train_metrics["train_samples_per_second"], 0.5)

        if do_eval:
            eval_metrics = load_json(os.path.join(output_dir, "eval_results.json"))
            self.assertIn("eval_bleu", eval_metrics)
            if quality_checks:
                self.assertGreater(eval_metrics["eval_bleu"], 1)

    # XXX: need to do better validation beyond just that the run was successful
    def run_and_check(
        self,
        stage,
        model_name: str = T5_SMALL,
        eval_steps: int = 10,
        distributed: bool = True,
        do_train: bool = True,
        do_eval: bool = True,
        quality_checks: bool = True,
        fp16: bool = True,
        extra_args_str: str = None,
        remove_args_str: str = None,
    ):

        # we are doing quality testing so using a small real model
        output_dir = self.run_trainer(
            stage=stage,
            model_name=model_name,
            eval_steps=eval_steps,
            num_train_epochs=1,
            do_train=do_train,
            do_eval=do_eval,
            distributed=distributed,
            fp16=fp16,
            extra_args_str=extra_args_str,
            remove_args_str=remove_args_str,
        )

        self.do_checks(output_dir, do_train=do_train, do_eval=do_eval, quality_checks=quality_checks)

        return output_dir

    def run_trainer(
        self,
        stage: str,
        model_name: str,
        eval_steps: int = 10,
        num_train_epochs: int = 1,
        do_train: bool = False,
        do_eval: bool = True,
        distributed: bool = True,
        fp16: bool = True,
        extra_args_str: str = None,
        remove_args_str: str = None,
    ):
        max_len = 32
        data_dir = self.test_file_dir / "../fixtures/tests_samples/wmt_en_ro"
        output_dir = self.get_auto_remove_tmp_dir()
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
            --save_steps 0
            --eval_steps {eval_steps}
            --group_by_length
            --label_smoothing_factor 0.1
            --source_lang en
            --target_lang ro
            --report_to none
        """.split()
        args.extend(["--source_prefix", '"translate English to Romanian: "'])

        if fp16:
            args.extend(["--fp16"])

        actions = 0
        if do_train:
            actions += 1
            args.extend(
                f"""
            --do_train
            --num_train_epochs {str(num_train_epochs)}
            --max_train_samples 16
            --per_device_train_batch_size 2
            --learning_rate 3e-3
            """.split()
            )

        if do_eval:
            actions += 1
            args.extend(
                """
            --do_eval
            --max_eval_samples 16
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

        ds_args = f"--deepspeed {self.test_file_dir_str}/ds_config_{stage}.json".split()
        script = [f"{self.examples_dir_str}/pytorch/translation/run_translation.py"]
        launcher = get_launcher(distributed)

        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        return output_dir

    @parameterized.expand(stages)
    def test_clm(self, stage):
        # this test exercises model.resize_token_embeddings() which requires param gathering outside
        # of forward - it's not used by `run_translation.py`, but it is in `run_clm.py`

        data_dir = self.tests_dir / "fixtures"
        output_dir = self.get_auto_remove_tmp_dir()
        args = f"""
            --model_name_or_path {GPT2_TINY}
            --train_file {data_dir}/sample_text.txt
            --validation_file {data_dir}/sample_text.txt
            --output_dir {output_dir}
            --overwrite_output_dir
            --do_train
            --do_eval
            --max_train_samples 16
            --max_eval_samples 16
            --per_device_train_batch_size 2
            --per_device_eval_batch_size 2
            --num_train_epochs 1
            --warmup_steps 8
            --block_size 64
            --fp16
            --report_to none
            """.split()

        ds_args = f"--deepspeed {self.test_file_dir_str}/ds_config_{stage}.json".split()
        script = [f"{self.examples_dir_str}/pytorch/language-modeling/run_clm.py"]
        launcher = get_launcher(distributed=True)

        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

    def test_clm_from_config_zero3(self):
        # this test exercises AutoModel.from_config(config) - to ensure zero.Init is called

        data_dir = self.tests_dir / "fixtures"
        output_dir = self.get_auto_remove_tmp_dir()
        args = f"""
            --model_type gpt2
            --tokenizer_name {GPT2_TINY}
            --train_file {data_dir}/sample_text.txt
            --validation_file {data_dir}/sample_text.txt
            --output_dir {output_dir}
            --overwrite_output_dir
            --do_train
            --max_train_samples 4
            --per_device_train_batch_size 2
            --num_train_epochs 1
            --warmup_steps 8
            --block_size 8
            --fp16
            --report_to none
            """.split()

        ds_args = f"--deepspeed {self.test_file_dir_str}/ds_config_zero3.json".split()
        script = [f"{self.examples_dir_str}/pytorch/language-modeling/run_clm.py"]
        launcher = get_launcher(distributed=True)

        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        with CaptureStderr() as cs:
            execute_subprocess_async(cmd, env=self.get_env())
        self.assertIn("Detected DeepSpeed ZeRO-3", cs.err)

    @parameterized.expand(stages)
    def test_load_best_model(self, stage):
        # this test exercises --load_best_model_at_end - the key is being able to resume after some training

        data_dir = self.tests_dir / "fixtures/tests_samples/wmt_en_ro"
        output_dir = self.get_auto_remove_tmp_dir()
        args = f"""
            --model_name_or_path {T5_TINY}
            --tokenizer_name {T5_TINY}
            --train_file {data_dir}/train.json
            --validation_file {data_dir}/val.json
            --output_dir {output_dir}
            --overwrite_output_dir
            --source_lang en
            --target_lang ro
            --do_train
            --max_train_samples 3
            --do_eval
            --max_eval_samples 1
            --logging_strategy steps
            --logging_steps 1
            --evaluation_strategy steps
            --eval_steps 1
            --save_strategy steps
            --save_steps 1
            --load_best_model_at_end
            --per_device_train_batch_size 1
            --per_device_eval_batch_size 1
            --num_train_epochs 1
            --fp16
            --report_to none
            """.split()
        args.extend(["--source_prefix", "translate English to Romanian: "])

        ds_args = f"--deepspeed {self.test_file_dir_str}/ds_config_{stage}.json".split()
        script = [f"{self.examples_dir_str}/pytorch/translation/run_translation.py"]
        launcher = get_launcher(distributed=False)

        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        with CaptureStd() as cs:
            execute_subprocess_async(cmd, env=self.get_env())
        # enough to test it didn't fail
        self.assertIn("DeepSpeed info", cs.out)
