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

"""
DeepSpeed-specific trainer tests.

Ported from tests/deepspeed/test_deepspeed.py.
Includes single-GPU integration tests and multi-GPU distributed tests.
"""

import dataclasses
import itertools
import json
import os
import subprocess
import unittest
from copy import deepcopy
from functools import partial

import datasets
from parameterized import parameterized

import transformers
from transformers import AutoModel, TrainingArguments, is_torch_available, logging
from transformers.integrations.deepspeed import (
    HfDeepSpeedConfig,
    is_deepspeed_available,
    unset_hf_deepspeed_config,
)
from transformers.testing_utils import (
    CaptureLogger,
    CaptureStd,
    LoggingLevel,
    TestCasePlus,
    backend_device_count,
    execute_subprocess_async,
    get_tests_dir,
    get_torch_dist_unique_port,
    mockenv_context,
    read_json_file,
    require_deepspeed,
    require_optuna,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)
from transformers.trainer_utils import get_last_checkpoint, set_seed
from transformers.utils import SAFE_WEIGHTS_NAME, is_torch_bf16_available_on_device, is_torch_fp16_available_on_device

from .test_trainer_distributed import CONFIGS_DIR, SCRIPTS_DIR, TRAIN_SCRIPT, TrainerDistributedCommon


if is_torch_available():
    import torch
    import torch.nn as nn

    from tests.trainer.trainer_test_utils import (  # noqa
        RegressionModelConfig,
        RegressionPreTrainedModel,
    )
    from tests.trainer.trainer_test_utils import (
        get_regression_trainer as _get_regression_trainer,
    )

    # hack to restore original logging level pre #21700
    get_regression_trainer = partial(_get_regression_trainer, log_level="info")


set_seed(42)

T5_TINY = "patrickvonplaten/t5-tiny-random"
GPT2_TINY = "sshleifer/tiny-gpt2"
GPTJ_TINY = "hf-internal-testing/tiny-random-gptj"

# Accelerate config paths
DS_ZERO2_CONFIG_FILE = os.path.join(CONFIGS_DIR, "deepspeed_zero2.yaml")
DS_ZERO3_CONFIG_FILE = os.path.join(CONFIGS_DIR, "deepspeed_zero3.yaml")

# DS JSON config paths (used by single-GPU tests via dict and distributed tests via file)
DS_CONFIG_ZERO2 = os.path.join(SCRIPTS_DIR, "ds_config_zero2.json")
DS_CONFIG_ZERO3 = os.path.join(SCRIPTS_DIR, "ds_config_zero3.json")

DS_CONFIGS = {
    "zero2": DS_ZERO2_CONFIG_FILE,
    "zero3": DS_ZERO3_CONFIG_FILE,
}

DS_JSON_CONFIGS = {
    "zero2": DS_CONFIG_ZERO2,
    "zero3": DS_CONFIG_ZERO3,
}

ZERO2 = "zero2"
ZERO3 = "zero3"

FP16 = "fp16"
BF16 = "bf16"

HF_OPTIM = "hf_optim"
HF_SCHEDULER = "hf_scheduler"
DS_OPTIM = "ds_optim"
DS_SCHEDULER = "ds_scheduler"

optims = [HF_OPTIM, DS_OPTIM]
schedulers = [HF_SCHEDULER, DS_SCHEDULER]

stages = [ZERO2, ZERO3]

dtypes = []
if is_torch_bf16_available_on_device(torch_device):
    dtypes.append(BF16)
if is_torch_fp16_available_on_device(torch_device):
    dtypes.append(FP16)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def require_deepspeed_aio(test_case):
    """
    Decorator marking a test that requires deepspeed aio (nvme)
    """
    if not is_deepspeed_available():
        return unittest.skip(reason="test requires deepspeed")(test_case)

    import deepspeed
    from deepspeed.ops.aio import AsyncIOBuilder

    if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
        return unittest.skip(reason="test requires deepspeed async-io")(test_case)
    else:
        return test_case


if is_deepspeed_available():
    from deepspeed.utils import logger as deepspeed_logger  # noqa
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

    from transformers.integrations.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled  # noqa


def _parameterized_custom_name_func(func, param_num, param):
    param_based_name = parameterized.to_safe_name("_".join(str(x) for x in param.args))
    return f"{func.__name__}_{param_based_name}"


# Parameterized test combinations
# Pure dtype: model loaded in target dtype, no mixed precision flags
pure_dtype_params = list(itertools.product(stages, ["fp32"] + dtypes))
# Mixed precision: model loaded in fp32, training with --bf16/--fp16
mixed_precision_params = list(itertools.product(stages, dtypes))
params = mixed_precision_params  # alias used by single-GPU tests (stage x dtype)
params_with_optims_and_schedulers = list(itertools.product(stages, dtypes, optims, schedulers))


# ---------------------------------------------------------------------------
# Command mixin
# ---------------------------------------------------------------------------


class DeepSpeedCommandsMixin:
    """Provides ``get_torchrun_cmd`` and ``get_accelerate_cmd`` for DeepSpeed distributed tests."""

    def get_torchrun_cmd(self, script, script_args=None, num_processes=None):
        if num_processes is None:
            num_processes = backend_device_count(torch_device)
        port = get_torch_dist_unique_port()
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_processes}",
            "--nnodes=1",
            f"--master_port={port}",
            script,
        ]
        if script_args:
            cmd.extend(script_args)
        return cmd

    def get_accelerate_cmd(
        self, script, config_file, launch_args=None, script_args=None, num_processes=None, **kwargs
    ):
        if num_processes is None:
            num_processes = backend_device_count(torch_device)
        port = get_torch_dist_unique_port()
        cmd = [
            "accelerate",
            "launch",
            "--config_file",
            config_file,
            "--num_processes",
            str(num_processes),
            "--main_process_port",
            str(port),
        ]
        if launch_args:
            cmd.extend(launch_args)
        cmd.append(script)
        if script_args:
            cmd.extend(script_args)
        return cmd


# ---------------------------------------------------------------------------
# Single-GPU Trainer + DeepSpeed integration tests
# ---------------------------------------------------------------------------


@require_deepspeed
@require_torch_accelerator
class TestTrainerIntegrationDeepSpeed(TestCasePlus):
    """Single-GPU DeepSpeed + Trainer integration tests (runs in-process, not via launcher).

    These tests use the DeepSpeed JSON config (``DS_JSON_CONFIGS``) directly, **not**
    the Accelerate YAML config files, because they run on a single GPU without
    ``accelerate launch``.
    """

    def setUp(self):
        super().setUp()

        master_port = get_torch_dist_unique_port()
        self.dist_env_1_gpu = {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        }

    def tearDown(self):
        super().tearDown()

        # reset the ds config global so that tests state doesn't leak
        unset_hf_deepspeed_config()

    def get_config_dict(self, stage):
        """Load a fresh DS config dict (safe to mutate)."""
        return load_json(DS_JSON_CONFIGS[stage])

    def check_trainer_state_are_the_same(self, trainer_state, trainer_state1):
        state = trainer_state.copy()
        state1 = trainer_state1.copy()
        log_history = state.pop("log_history", None)
        log_history1 = state1.pop("log_history", None)
        self.assertEqual(state, state1)
        skip_log_keys = ["train_runtime", "train_samples_per_second", "train_steps_per_second", "train_loss"]
        for log, log1 in zip(log_history, log_history1):
            for key in skip_log_keys:
                _ = log.pop(key, None)
                _ = log1.pop(key, None)
            # Remove NaN entries
            for d in (log, log1):
                for key in list(d.keys()):
                    if d[key] != d[key]:
                        del d[key]
            self.assertEqual(log, log1)

    # --- Deepspeed config tests --- #

    def test_hf_ds_config_mismatch(self):
        """Verify that mismatched DS config and TrainingArguments raises with all conflicting keys."""
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
                fp16=fp16,
                deepspeed=ds_config,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                adam_beta1=adam_beta1,
                adam_beta2=adam_beta2,
                output_dir=self.get_auto_remove_tmp_dir(),
            )
            with self.assertRaises(Exception) as context:
                trainer.train()

        for key in keys:
            self.assertTrue(
                key in str(context.exception),
                f"{key} is not in the exception message:\n{context.exception}",
            )

    def test_deepspeed_plugin_from_config(self):
        """TrainingArguments(deepspeed=config) creates a plugin with correct zero_stage."""
        for stage, expected in [(ZERO2, 2), (ZERO3, 3)]:
            with mockenv_context(**self.dist_env_1_gpu):
                args = TrainingArguments(
                    deepspeed=self.get_config_dict(stage),
                    output_dir=self.get_auto_remove_tmp_dir(),
                )
                self.assertIsNotNone(args.deepspeed_plugin)
                self.assertEqual(args.deepspeed_plugin.zero_stage, expected)

    def test_auto_value_resolution(self):
        """Verify 'auto' placeholders in DS config are replaced with TrainingArguments values."""
        lr = 3e-4
        batch_size = 4
        grad_acc = 2

        ds_config = self.get_config_dict(ZERO2)

        with mockenv_context(**self.dist_env_1_gpu):
            args = TrainingArguments(
                deepspeed=ds_config,
                learning_rate=lr,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_acc,
                bf16=True,
                output_dir=self.get_auto_remove_tmp_dir(),
            )
            config = args.hf_deepspeed_config.config
            self.assertEqual(config["train_micro_batch_size_per_gpu"], batch_size)
            self.assertEqual(config["gradient_accumulation_steps"], grad_acc)
            self.assertEqual(config["train_batch_size"], batch_size * grad_acc)
            self.assertEqual(config["optimizer"]["params"]["lr"], lr)
            self.assertTrue(config["bf16"]["enabled"])
            self.assertFalse(config["fp16"]["enabled"])

    def test_ds_config_dtype_resolution(self):
        """hf_deepspeed_config.dtype() returns the correct torch dtype."""
        test_cases = [
            ({"bf16": True}, torch.bfloat16),
            ({"fp16": True}, torch.float16),
            ({}, torch.float32),
        ]
        for kwargs, expected_dtype in test_cases:
            ds_config = self.get_config_dict(ZERO2)
            with mockenv_context(**self.dist_env_1_gpu):
                args = TrainingArguments(
                    deepspeed=ds_config,
                    output_dir=self.get_auto_remove_tmp_dir(),
                    **kwargs,
                )
                self.assertEqual(args.hf_deepspeed_config.dtype(), expected_dtype)

    def test_ds_config_file_path_vs_dict_equivalent(self):
        """Passing DS config as file path or dict produces equivalent resolved configs."""
        with mockenv_context(**self.dist_env_1_gpu):
            args_dict = TrainingArguments(
                deepspeed=self.get_config_dict(ZERO3),
                output_dir=self.get_auto_remove_tmp_dir(),
            )
            args_file = TrainingArguments(
                deepspeed=DS_JSON_CONFIGS[ZERO3],
                output_dir=self.get_auto_remove_tmp_dir(),
            )
            self.assertEqual(args_dict.hf_deepspeed_config.config, args_file.hf_deepspeed_config.config)
            self.assertEqual(args_dict.deepspeed_plugin.zero_stage, args_file.deepspeed_plugin.zero_stage)

    def test_ds_config_no_optimizer_no_scheduler(self):
        """When optimizer/scheduler are removed from DS config, HF defaults kick in."""
        ds_config = self.get_config_dict(ZERO2)
        del ds_config["optimizer"]
        del ds_config["scheduler"]

        with mockenv_context(**self.dist_env_1_gpu):
            args = TrainingArguments(
                deepspeed=ds_config,
                output_dir=self.get_auto_remove_tmp_dir(),
            )
            self.assertIsNotNone(args.deepspeed_plugin)
            resolved = args.hf_deepspeed_config.config
            self.assertNotIn("optimizer", resolved)
            self.assertNotIn("scheduler", resolved)

            # Verify training still works (HF defaults for optimizer/scheduler)
            trainer = get_regression_trainer(deepspeed=ds_config, fp16=True, output_dir=self.get_auto_remove_tmp_dir())
            trainer.train()

    def test_ds_config_auto_vs_missing_fields(self):
        """Fields set to 'auto' are resolved from TrainingArguments; absent fields are not filled by HF."""
        lr = 5e-5
        batch_size = 4
        grad_acc = 2

        # Config with "auto" placeholders — values should be filled from TrainingArguments
        ds_config_auto = self.get_config_dict(ZERO2)

        # Config with top-level "auto" fields removed entirely
        ds_config_missing = self.get_config_dict(ZERO2)
        del ds_config_missing["train_micro_batch_size_per_gpu"]
        del ds_config_missing["gradient_accumulation_steps"]
        del ds_config_missing["train_batch_size"]
        del ds_config_missing["gradient_clipping"]

        common_kwargs = {
            "learning_rate": lr,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": grad_acc,
        }

        with mockenv_context(**self.dist_env_1_gpu):
            # "auto" config: fields should be resolved to match TrainingArguments
            args_auto = TrainingArguments(
                deepspeed=ds_config_auto, output_dir=self.get_auto_remove_tmp_dir(), **common_kwargs
            )
            cfg_auto = args_auto.hf_deepspeed_config.config
            self.assertEqual(cfg_auto["train_micro_batch_size_per_gpu"], batch_size)
            self.assertEqual(cfg_auto["gradient_accumulation_steps"], grad_acc)
            self.assertEqual(cfg_auto["optimizer"]["params"]["lr"], lr)

            # missing config: HF's fill_match is a no-op, so fields are NOT resolved
            # from TrainingArguments. Note: accelerate/deepspeed may inject its own
            # defaults (e.g. gradient_accumulation_steps=1), but those won't match our
            # TrainingArguments values.
            args_missing = TrainingArguments(
                deepspeed=ds_config_missing, output_dir=self.get_auto_remove_tmp_dir(), **common_kwargs
            )
            cfg_missing = args_missing.hf_deepspeed_config.config
            self.assertNotIn("train_micro_batch_size_per_gpu", cfg_missing)
            self.assertNotIn("train_batch_size", cfg_missing)
            self.assertNotIn("gradient_clipping", cfg_missing)
            # gradient_accumulation_steps may be injected by accelerate with a default,
            # but it should NOT be resolved to our TrainingArguments value
            if "gradient_accumulation_steps" in cfg_missing:
                self.assertNotEqual(cfg_missing["gradient_accumulation_steps"], grad_acc)

            # Plugin should still be created successfully in both cases
            self.assertIsNotNone(args_auto.deepspeed_plugin)
            self.assertIsNotNone(args_missing.deepspeed_plugin)

    @parameterized.expand(stages, name_func=_parameterized_custom_name_func)
    def test_mixed_precision_model_and_optimizer_dtypes(self, stage):
        """DeepSpeed bf16 keeps model params in bf16 and optimizer master weights in fp32."""
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(
                bf16=True, deepspeed=self.get_config_dict(stage), output_dir=self.get_auto_remove_tmp_dir()
            )
            trainer.train()

            # Model params should be in bf16 (used for forward/backward)
            for param in trainer.model.parameters():
                self.assertEqual(param.dtype, torch.bfloat16, "Model params should be bf16 during DS bf16 training")

            # Optimizer master weights should be in fp32. DeepSpeed always maintains
            # fp32 copies for the optimizer step, regardless of model dtype.
            ds_optimizer = trainer.model_wrapped.optimizer
            # ZeRO-2: single_partition_of_fp32_groups
            # ZeRO-3: fp32_partitioned_groups_flat (list of lists)
            if hasattr(ds_optimizer, "single_partition_of_fp32_groups"):
                master_weights = ds_optimizer.single_partition_of_fp32_groups
            else:
                master_weights = [p for group in ds_optimizer.fp32_partitioned_groups_flat for p in group]
            for partition in master_weights:
                self.assertEqual(partition.dtype, torch.float32, "Optimizer master weights should be fp32")

    def test_ds_config_object(self):
        """Switching between ZeRO-2 and ZeRO-3 configs in one process works correctly."""
        output_dir = self.get_auto_remove_tmp_dir()
        kwargs = {"output_dir": output_dir, "train_len": 8, "fp16": True}

        ds_config_zero3_dict = self.get_config_dict(ZERO3)
        ds_config_zero2_dict = self.get_config_dict(ZERO2)

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

            # with accelerate integration below line is additionally required for this test to pass
            trainer.accelerator.state._reset_state()
            del trainer
            # now weakref should gc the global and we shouldn't get anything here
            config = deepspeed_config()
            self.assertFalse(is_deepspeed_zero3_enabled())
            self.assertFalse(bool(config), "Deepspeed config should not be accessible")

    # --- Integration tests --- #
    @parameterized.expand(params, name_func=_parameterized_custom_name_func)
    def test_basic_training(self, stage, dtype):
        """Train with default DS config (no dict modifications)."""
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(
                deepspeed=DS_JSON_CONFIGS[stage],
                output_dir=self.get_auto_remove_tmp_dir(),
                **{dtype: True},
            )
            with CaptureLogger(deepspeed_logger) as cl:
                trainer.train()
            self.assertIn("DeepSpeed info", cl.out, "expected DeepSpeed logger output but got none")

    @parameterized.expand(
        [
            ("hf_optim_hf_scheduler", True, True),
            ("hf_optim_ds_scheduler", True, False),
            ("ds_optim_hf_scheduler", False, True),
            ("ds_optim_ds_scheduler", False, False),
        ],
        name_func=_parameterized_custom_name_func,
    )
    def test_optimizer_scheduler_combos(self, _, use_hf_optim, use_hf_scheduler):
        """Verify non-default optimizer/scheduler combos train successfully."""
        a = 0
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero2_dict = self.get_config_dict(ZERO2)
            if use_hf_optim:
                del ds_config_zero2_dict["optimizer"]
            if use_hf_scheduler:
                del ds_config_zero2_dict["scheduler"]
            ds_config_zero2_dict["zero_optimization"]["offload_optimizer"] = {"device": "none"}
            trainer = get_regression_trainer(
                a=a, bf16=True, deepspeed=ds_config_zero2_dict, output_dir=self.get_auto_remove_tmp_dir()
            )
            trainer.train()
        new_a = trainer.model.a.item()
        self.assertNotEqual(new_a, a)

    @parameterized.expand(params, name_func=_parameterized_custom_name_func)
    def test_gradient_accumulation(self, stage, dtype):
        """Check that gradient accumulation produces the same weights as a larger batch."""
        train_len = 64
        a = b = 0.0

        kwargs = {
            "a": a,
            "b": b,
            "train_len": train_len,
            "deepspeed": DS_JSON_CONFIGS[stage],
            "output_dir": self.get_auto_remove_tmp_dir(),
            dtype: True,
        }

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
        # weights, but sometimes get a slight difference still
        self.assertAlmostEqual(no_grad_accum_a, yes_grad_accum_a, delta=1e-4)
        self.assertAlmostEqual(no_grad_accum_b, yes_grad_accum_b, delta=1e-4)

        # Relative difference
        self.assertTrue((no_grad_accum_loss - yes_grad_accum_loss) / (no_grad_accum_loss + 1e-15) <= 1e-3)

    @parameterized.expand(
        list(itertools.product(stages, [True, False])),
        name_func=_parameterized_custom_name_func,
    )
    def test_optimizer_with_cpu_offload(self, stage, use_hf_optim):
        """CPU offload works with both HF and DS optimizers."""
        ds_config_dict = self.get_config_dict(stage)
        ds_config_dict["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}
        if use_hf_optim:
            del ds_config_dict["optimizer"]
            # By default, accelerate swaps the HF optimizer for DeepSpeed's CPUAdam when
            # CPU offload is enabled (since offload is inefficient without it). Setting this
            # to False keeps the original HF optimizer, which is what we want to test here.
            ds_config_dict["zero_force_ds_cpu_optimizer"] = False
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(
                deepspeed=ds_config_dict, bf16=True, output_dir=self.get_auto_remove_tmp_dir()
            )
            with CaptureLogger(deepspeed_logger) as cl:
                trainer.train()
            self.assertIn("DeepSpeed info", cl.out, "expected DeepSpeed logger output but got none")

    @require_deepspeed_aio
    def test_stage3_nvme_offload(self):
        """Train with ZeRO-3 NVMe offload for both optimizer and params."""
        with mockenv_context(**self.dist_env_1_gpu):
            # this actually doesn't have to be on NVMe, any storage will do since this test only
            # runs a simple check that we can use some directory as if it were NVMe
            nvme_path = self.get_auto_remove_tmp_dir()
            nvme_config = {"device": "nvme", "nvme_path": nvme_path}
            ds_config_zero3_dict = self.get_config_dict(ZERO3)
            ds_config_zero3_dict["zero_optimization"]["offload_optimizer"] = nvme_config
            ds_config_zero3_dict["zero_optimization"]["offload_param"] = nvme_config
            ds_config_zero3_dict["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True
            trainer = get_regression_trainer(
                fp16=True, deepspeed=ds_config_zero3_dict, output_dir=self.get_auto_remove_tmp_dir()
            )
            with CaptureLogger(deepspeed_logger) as cl:
                trainer.train()
            self.assertIn("DeepSpeed info", cl.out, "expected DeepSpeed logger output but got none")

    @parameterized.expand(params, name_func=_parameterized_custom_name_func)
    def test_early_get_last_lr(self, stage, dtype):
        """Ensure get_last_lr() doesn't crash on the very first logging step."""
        with mockenv_context(**self.dist_env_1_gpu):
            a = b = 0.0
            trainer = get_regression_trainer(
                a=a,
                b=b,
                train_len=8,
                deepspeed=DS_JSON_CONFIGS[stage],
                per_device_train_batch_size=8,
                logging_steps=1,
                output_dir=self.get_auto_remove_tmp_dir(),
                **{dtype: True},
            )

            trainer.train()
            post_train_a = trainer.model.a.item()

            # XXX: for some reason the following check fails with zero3/fp16 and any/bf16 - not a
            # broken but a different qualitative outcome - as if optimizer did run
            if (stage == ZERO3 and dtype == FP16) or (dtype == BF16):
                self.skipTest(reason="When using zero3/fp16 or any/bf16 the optimizer seems run oddly")

            # it's enough that train didn't fail for this test, but we must check that
            # optimizer/scheduler didn't run (since if it did this test isn't testing the right thing)
            self.assertEqual(post_train_a, a)

    def check_saved_checkpoints_deepspeed(self, output_dir, freq, total, stage, dtype):
        # adapted from TrainerIntegrationCommon.check_saved_checkpoints
        file_list = [SAFE_WEIGHTS_NAME, "training_args.bin", "trainer_state.json", "config.json"]

        # DeepSpeed checkpoint file names differ by ZeRO stage:
        # - ZeRO-2: params are not partitioned, so the file is simply mp_rank_00_model_states.pt
        # - ZeRO-3: params are partitioned across ranks, using the zero_pp_ prefix
        if stage == ZERO2:
            ds_file_list = ["mp_rank_00_model_states.pt"]
        elif stage == ZERO3:
            ds_file_list = ["zero_pp_rank_0_mp_rank_00_model_states.pt"]
        else:
            raise ValueError(f"unknown stage {stage}")

        # With bf16, DeepSpeed uses BF16_Optimizer which stores optimizer states (Adam
        # momentum/variance) in a separate file with a bf16_ prefix. With fp16, optimizer
        # states are bundled inside the model states file.
        if dtype == "bf16":
            ds_file_list.append("bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt")

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
                path = os.path.join(ds_path, filename)
                self.assertTrue(os.path.isfile(path), f"[{stage}] {path} is not found")

    @parameterized.expand(params, name_func=_parameterized_custom_name_func)
    def test_save_checkpoints(self, stage, dtype):
        """Verify DS checkpoint file structure at each save step."""
        freq = 5
        output_dir = self.get_auto_remove_tmp_dir()
        ds_config_dict = self.get_config_dict(stage)
        if stage == ZERO3:
            # ZeRO-3 partitions params across ranks; without this flag only sharded DS
            # state files are saved. We need it so a consolidated model.safetensors is
            # written, which check_saved_checkpoints_deepspeed verifies.
            ds_config_dict["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True

        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(
                output_dir=output_dir, save_steps=freq, deepspeed=ds_config_dict, **{dtype: True}
            )
            trainer.train()

        total = int(3.0 * 64 / 8)  # n_epochs * train_len / per_device_train_batch_size
        self.check_saved_checkpoints_deepspeed(output_dir, freq, total, stage, dtype)

        # Verify we can resume training from the last checkpoint with a new trainer
        checkpoint = get_last_checkpoint(output_dir)
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(
                output_dir=output_dir, save_steps=freq, deepspeed=ds_config_dict, **{dtype: True}
            )
            trainer.train(resume_from_checkpoint=checkpoint)

    @parameterized.expand(params, name_func=_parameterized_custom_name_func)
    def test_can_resume_training_errors(self, stage, dtype):
        """Verify correct errors when resuming from missing/bogus checkpoints."""
        with mockenv_context(**self.dist_env_1_gpu):
            output_dir = self.get_auto_remove_tmp_dir()
            trainer = get_regression_trainer(output_dir=output_dir, deepspeed=DS_JSON_CONFIGS[stage], **{dtype: True})

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

    @parameterized.expand(params_with_optims_and_schedulers, name_func=_parameterized_custom_name_func)
    def test_can_resume_training_normal(self, stage, dtype, optim, scheduler):
        """Resume training from checkpoint and verify weights/state match a full run."""

        # ToDo: Currently, hf_optim + hf_scheduler resumes with the correct states and
        # also has same losses for few steps but then slowly diverges. Need to figure it out.
        if optim == HF_OPTIM and scheduler == HF_SCHEDULER:
            self.skipTest(reason="hf_optim + hf_scheduler resumes with the correct states but slowly diverges")

        output_dir = self.get_auto_remove_tmp_dir()
        ds_config_dict = self.get_config_dict(stage)
        if dtype == FP16:
            ds_config_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
        if stage == ZERO3:
            ds_config_dict["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True

        if optim == HF_OPTIM:
            del ds_config_dict["optimizer"]

        if scheduler == HF_SCHEDULER:
            del ds_config_dict["scheduler"]

        kwargs = {
            "output_dir": output_dir,
            "train_len": 128,
            "save_steps": 5,
            "learning_rate": 0.1,
            "deepspeed": ds_config_dict,
            dtype: True,
        }

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
            # XXX: but currently this not possible due DS bug: https://github.com/deepspeedai/DeepSpeed/issues/1612
            # DummyScheduler lacks load_state_dict, so this fails.
            # trainer.train(resume_from_checkpoint=checkpoint)

    @parameterized.expand(params, name_func=_parameterized_custom_name_func)
    def test_load_state_dict_from_zero_checkpoint(self, stage, dtype):
        """Load fp32 weights from a zero checkpoint and verify they match the trained model."""
        output_dir = self.get_auto_remove_tmp_dir()

        kwargs = {
            "output_dir": output_dir,
            "train_len": 4,
            "per_device_train_batch_size": 4,
            "num_train_epochs": 1,
            "save_strategy": "steps",
            "save_steps": 1,
            "learning_rate": 0.1,
            "deepspeed": DS_JSON_CONFIGS[stage],
            dtype: True,
        }

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

    @parameterized.expand(stages, name_func=_parameterized_custom_name_func)
    def test_load_best_model(self, stage):
        """load_best_model_at_end re-initializes the DS engine; verify train + eval don't crash."""
        from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer

        output_dir = self.get_auto_remove_tmp_dir()

        ds_config_dict = self.get_config_dict(stage)
        del ds_config_dict["optimizer"]
        del ds_config_dict["scheduler"]
        if stage == ZERO3:
            ds_config_dict["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True

        with mockenv_context(**self.dist_env_1_gpu):
            training_args = TrainingArguments(
                output_dir,
                max_steps=1,
                eval_strategy="steps",
                eval_steps=1,
                save_steps=1,
                load_best_model_at_end=True,
                bf16=True,
                deepspeed=ds_config_dict,
            )

            tokenizer = T5Tokenizer.from_pretrained(T5_TINY)
            model = T5ForConditionalGeneration.from_pretrained(T5_TINY)

            data_file = str(self.tests_dir / "fixtures/tests_samples/SQUAD/sample.json")
            raw_datasets = datasets.load_dataset("json", data_files=data_file, field="data")
            train_dataset = (
                raw_datasets["train"]
                .map(
                    lambda ex: {
                        "input_text": f"question: {ex['question']}  context: {ex['context']}",
                        "target_text": ex["answers"]["text"][0] if ex["answers"]["text"] else "",
                    }
                )
                .map(
                    lambda batch: {
                        "input_ids": tokenizer(
                            batch["input_text"], padding="max_length", max_length=512, truncation=True
                        )["input_ids"],
                        "attention_mask": tokenizer(
                            batch["input_text"], padding="max_length", max_length=512, truncation=True
                        )["attention_mask"],
                        "labels": tokenizer(
                            batch["target_text"], padding="max_length", max_length=16, truncation=True
                        )["input_ids"],
                    },
                    batched=True,
                )
            )
            eval_dataset = deepcopy(train_dataset)

            trainer = Trainer(
                model=model,
                processing_class=tokenizer,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
            trainer.train()
            trainer.evaluate()

    @require_optuna
    def test_hyperparameter_search(self):
        """Run Optuna hyperparameter search with DeepSpeed ZeRO-3."""
        with mockenv_context(**self.dist_env_1_gpu):
            # hyperparameter_search requires model_init() to recreate the model for each trial
            def model_init():
                config = RegressionModelConfig(a=0, b=0, double_output=False)
                model = RegressionPreTrainedModel(config)
                return model

            trainer = get_regression_trainer(
                fp16=True,
                model_init=model_init,
                deepspeed=DS_JSON_CONFIGS[ZERO3],
                output_dir=self.get_auto_remove_tmp_dir(),
            )

            n_trials = 3
            with CaptureLogger(deepspeed_logger) as cl:
                with CaptureStd() as cs:
                    trainer.hyperparameter_search(direction="maximize", n_trials=n_trials)
            self.assertIn("DeepSpeed info", cl.out, "expected DeepSpeed logger output but got none")
            self.assertIn(f"Trial {n_trials - 1} finished with value", cs.err, "expected hyperparameter_search output")
            self.assertIn("Best is trial", cs.err, "expected hyperparameter_search output")


# ---------------------------------------------------------------------------
# DeepSpeed distributed tests
# ---------------------------------------------------------------------------


@slow
@require_deepspeed
@require_torch_multi_accelerator
class TestTrainerDistributedDeepSpeed(DeepSpeedCommandsMixin, TestCasePlus):
    def _run_env_check(self, cmd, num_processes):
        """Run the env check script and return per-rank results."""
        execute_subprocess_async(cmd, env=self.get_env())
        output_dir = cmd[cmd.index("--output_dir") + 1]
        results = []
        for rank in range(num_processes):
            with open(os.path.join(output_dir, f"env_rank{rank}.json")) as f:
                results.append(json.load(f))
        return results

    def test_torchrun_accelerate_deepspeed_zero2_env_parity(self):
        """Verify torchrun+--deepspeed and accelerate launch produce the same DeepSpeed ZeRO-2 env."""
        script = os.path.join(SCRIPTS_DIR, "torchrun_env_check.py")
        num_processes = backend_device_count(torch_device)

        torchrun_dir = self.get_auto_remove_tmp_dir()
        torchrun_results = self._run_env_check(
            self.get_torchrun_cmd(
                script,
                script_args=["--output_dir", torchrun_dir, "--deepspeed", DS_CONFIG_ZERO2],
                num_processes=num_processes,
            ),
            num_processes,
        )

        accel_dir = self.get_auto_remove_tmp_dir()
        accel_results = self._run_env_check(
            self.get_accelerate_cmd(
                script, DS_ZERO2_CONFIG_FILE, script_args=["--output_dir", accel_dir], num_processes=num_processes
            ),
            num_processes,
        )

        self._check_parity(torchrun_results, accel_results, num_processes, expected_zero_stage=2)

    def test_torchrun_accelerate_deepspeed_zero3_env_parity(self):
        """Verify torchrun+--deepspeed and accelerate launch produce the same DeepSpeed ZeRO-3 env."""
        script = os.path.join(SCRIPTS_DIR, "torchrun_env_check.py")
        num_processes = backend_device_count(torch_device)

        torchrun_dir = self.get_auto_remove_tmp_dir()
        torchrun_results = self._run_env_check(
            self.get_torchrun_cmd(
                script,
                script_args=["--output_dir", torchrun_dir, "--deepspeed", DS_CONFIG_ZERO3],
                num_processes=num_processes,
            ),
            num_processes,
        )

        accel_dir = self.get_auto_remove_tmp_dir()
        accel_results = self._run_env_check(
            self.get_accelerate_cmd(
                script, DS_ZERO3_CONFIG_FILE, script_args=["--output_dir", accel_dir], num_processes=num_processes
            ),
            num_processes,
        )

        self._check_parity(torchrun_results, accel_results, num_processes, expected_zero_stage=3)

    def _check_parity(self, torchrun_results, accel_results, num_processes, expected_zero_stage):
        for rank in range(num_processes):
            tr, ac = torchrun_results[rank], accel_results[rank]

            # Both should agree on distributed env
            self.assertEqual(tr["args_world_size"], ac["args_world_size"])
            self.assertEqual(tr["args_process_index"], ac["args_process_index"])
            self.assertEqual(tr["args_parallel_mode"], ac["args_parallel_mode"])
            self.assertEqual(tr["accelerator_num_processes"], ac["accelerator_num_processes"])
            self.assertEqual(tr["accelerator_use_distributed"], ac["accelerator_use_distributed"])

            for info in (tr, ac):
                # Rank consistency across all layers
                self.assertEqual(info["env_world_size"], str(num_processes))
                self.assertEqual(info["env_rank"], str(rank))
                self.assertEqual(info["args_process_index"], rank)
                self.assertEqual(info["args_local_process_index"], rank)
                self.assertEqual(info["accelerator_process_index"], rank)
                self.assertEqual(info["accelerator_local_process_index"], rank)
                self.assertEqual(info["args_n_gpu"], 1)
                self.assertEqual(info["accelerator_is_main_process"], rank == 0)
                self.assertEqual(info["accelerator_is_local_main_process"], rank == 0)
                self.assertIn(f"cuda:{rank}", info["accelerator_device"])

                # Both should have DeepSpeed enabled with the correct stage
                self.assertEqual(info["accelerator_distributed_type"], "DistributedType.DEEPSPEED")
                self.assertTrue(info["trainer_is_deepspeed_enabled"])
                self.assertFalse(info["trainer_is_fsdp_enabled"])
                self.assertEqual(info["deepspeed_zero_stage"], expected_zero_stage)
                self.assertEqual(info["deepspeed_offload_optimizer_device"], "none")
                self.assertEqual(info["deepspeed_offload_param_device"], "none")
                self.assertNotIn("fsdp_version", info)


# ---------------------------------------------------------------------------
# Multi-GPU distributed DeepSpeed training tests
# ---------------------------------------------------------------------------


@slow
@require_deepspeed
@require_torch_multi_accelerator
class TestTrainerDistributedDeepSpeedCommon(DeepSpeedCommandsMixin, TrainerDistributedCommon, TestCasePlus):
    """
    Distributed DeepSpeed tests using ``accelerate launch``.

    Some tests use a simple training script (train.py), others use example
    scripts (run_translation.py, run_clm.py) for broader integration coverage.
    """

    # -------------------------------------------------------------------
    # Basic training
    # -------------------------------------------------------------------

    # Pure dtype training: model loaded in target dtype, no mixed precision.
    @parameterized.expand(pure_dtype_params, name_func=_parameterized_custom_name_func)
    def test_training(self, stage, model_dtype):
        self.check_training(model_dtype, config_file=DS_CONFIGS[stage])

    # Mixed precision training: model loaded in fp32, training in fp16/bf16.
    @parameterized.expand(mixed_precision_params, name_func=_parameterized_custom_name_func)
    def test_training_mixed_precision(self, stage, dtype):
        self.check_mixed_precision(dtype, config_file=DS_CONFIGS[stage])

    @parameterized.expand(stages, name_func=_parameterized_custom_name_func)
    def test_training_with_gradient_accumulation(self, stage):
        self.check_gradient_accumulation(config_file=DS_CONFIGS[stage])

    @parameterized.expand(stages, name_func=_parameterized_custom_name_func)
    def test_training_and_can_resume_normally(self, stage):
        self.check_resume(config_file=DS_CONFIGS[stage])

    @parameterized.expand(
        [
            (ZERO2, False),
            (ZERO3, False),
            (ZERO3, True),  # offload_param only supported on ZeRO-3
        ],
        name_func=_parameterized_custom_name_func,
    )
    def test_basic_run_with_cpu_offload(self, stage, offload_param):
        output_dir = self.get_auto_remove_tmp_dir()
        args = self._get_default_script_args(output_dir) + ["--bf16", "--max_steps", "10"]
        launch_args = ["--offload_optimizer_device", "cpu"]
        if offload_param:
            launch_args += ["--offload_param_device", "cpu"]
        execute_subprocess_async(
            self.get_accelerate_cmd(
                TRAIN_SCRIPT, script_args=args, config_file=DS_CONFIGS[stage], launch_args=launch_args
            ),
            env=self.get_env(),
        )

    def test_eval(self):
        # ZeRO inference only works with ZeRO-3
        self.check_eval(config_file=DS_CONFIGS[ZERO3])

    def test_alst_ulysses_sp(self):
        """Test that ALST/Ulysses sequence parallelism produces the same losses as without it."""
        sp_config = os.path.join(CONFIGS_DIR, "deepspeed_zero2_sp.yaml")
        common_args = [
            "--max_steps",
            "10",
            "--per_device_train_batch_size",
            "1",
            "--gradient_accumulation_steps",
            "1",
            "--logging_steps",
            "1",
            "--seed",
            "42",
            "--attn_implementation",
            "sdpa",
            "--pad_to_multiple_of",
            "4",
        ]

        # Step 1: Run with SP enabled
        sp_yes_dir = self.get_auto_remove_tmp_dir()
        sp_yes_losses = os.path.join(sp_yes_dir, "losses.json")
        sp_yes_eval = os.path.join(sp_yes_dir, "eval_metrics.json")
        cmd = self.get_accelerate_cmd(
            TRAIN_SCRIPT,
            config_file=sp_config,
            script_args=common_args
            + [
                "--output_dir",
                sp_yes_dir,
                "--per_device_eval_batch_size",
                "1",
                "--loss_output_file",
                sp_yes_losses,
                "--eval_output_file",
                sp_yes_eval,
            ],
        )
        execute_subprocess_async(cmd, env=self.get_env())

        # Step 2: Run without SP
        sp_no_dir = self.get_auto_remove_tmp_dir()
        sp_no_losses = os.path.join(sp_no_dir, "losses.json")
        cmd = self.get_accelerate_cmd(
            TRAIN_SCRIPT,
            config_file=DS_CONFIGS[ZERO2],
            script_args=common_args
            + [
                "--output_dir",
                sp_no_dir,
                "--loss_output_file",
                sp_no_losses,
            ],
        )
        execute_subprocess_async(cmd, env=self.get_env())

        # Compare losses — SP splits sequences across GPUs so expect some numerical divergence
        sp_yes_losses_data = read_json_file(sp_yes_losses)
        sp_no_losses_data = read_json_file(sp_no_losses)
        self.assertEqual(len(sp_yes_losses_data), len(sp_no_losses_data))
        torch.testing.assert_close(
            torch.tensor(sp_yes_losses_data),
            torch.tensor(sp_no_losses_data),
            atol=0.5,
            rtol=0.05,
        )

        # Eval should succeed even though eval sequences are not divisible by sp_size,
        # because SP is disabled in eval mode.
        eval_metrics = read_json_file(sp_yes_eval)
        self.assertIn("eval_loss", eval_metrics)
        self.assertTrue(torch.isfinite(torch.tensor(eval_metrics["eval_loss"])))


# ---------------------------------------------------------------------------
# Non-Trainer DeepSpeed integration tests (single GPU)
# ---------------------------------------------------------------------------


@require_deepspeed
@require_torch_accelerator
class TestNonTrainerIntegrationDeepSpeed(TestCasePlus):
    """
    Testing non-Trainer DeepSpeed integration
    """

    def setUp(self):
        super().setUp()

        master_port = get_torch_dist_unique_port()
        self.dist_env_1_gpu = {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        }

    def tearDown(self):
        super().tearDown()

        # reset the ds config global so that tests state doesn't leak
        unset_hf_deepspeed_config()

    def _get_zero3_ds_config(self, **extra):
        config = {
            "train_batch_size": 1,
            "zero_optimization": {"stage": 3},
        }
        config.update(extra)
        return config

    def _load_with_logging(self, model_cls, model_name, expect_zero3=True, **kwargs):
        """Load a pretrained model under mockenv, assert ZeRO-3 detection logging."""
        with LoggingLevel(logging.INFO):
            with mockenv_context(**self.dist_env_1_gpu):
                logger = logging.get_logger("transformers.modeling_utils")
                with CaptureLogger(logger) as cl:
                    result = model_cls.from_pretrained(model_name, **kwargs)
        if expect_zero3:
            self.assertIn("Detected DeepSpeed ZeRO-3", cl.out)
        else:
            self.assertNotIn("Detected DeepSpeed ZeRO-3", cl.out)
        return result, cl

    def _check_zero3_init_and_removal(self, extra_ds_config=None):
        """Test zero3 detection, then verify it stops after removing zero_optimization.

        Returns the model loaded under zero3 for further inspection.
        """
        ds_config = self._get_zero3_ds_config(**(extra_ds_config or {}))

        dschf = HfDeepSpeedConfig(ds_config)
        self.assertTrue(dschf.is_zero3())
        self.assertTrue(is_deepspeed_zero3_enabled())
        model, _ = self._load_with_logging(AutoModel, T5_TINY, expect_zero3=True)

        del ds_config["zero_optimization"]
        dschf = HfDeepSpeedConfig(ds_config)
        self.assertFalse(dschf.is_zero3())
        self.assertFalse(is_deepspeed_zero3_enabled())
        self._load_with_logging(AutoModel, T5_TINY, expect_zero3=False)

        return model

    @parameterized.expand(["default", "fp16", "bf16"], name_func=_parameterized_custom_name_func)
    def test_init_zero3(self, dtype):
        if dtype == "fp16" and not is_torch_fp16_available_on_device(torch_device):
            self.skipTest("test requires fp16 hardware support")
        if dtype == "bf16" and not is_torch_bf16_available_on_device(torch_device):
            self.skipTest("test requires bf16 hardware support")

        extra = {dtype: {"enabled": True}} if dtype in ("fp16", "bf16") else None
        model = self._check_zero3_init_and_removal(extra)

        # ZeRO-3 is the only stage that casts model weights to fp16/bf16 via zero.Init().
        # ZeRO-2/1 keep the model in its original dtype (fp32) and create half-precision
        # copies for forward/backward, meaning both fp32 and fp16/bf16 coexist in memory.
        if dtype == "fp16":
            expected_dtype = torch.float16
        elif dtype == "bf16":
            expected_dtype = torch.bfloat16
        else:
            expected_dtype = torch.float32

        for name, param in model.named_parameters():
            self.assertEqual(
                param.dtype, expected_dtype, f"Parameter {name} has dtype {param.dtype}, expected {expected_dtype}"
            )

    @require_torch_accelerator
    def test_from_config_zero3_weight_init(self):
        # test that from_config() correctly initializes weights under zero3
        # (regression test: without the fix, _init_weights runs on partitioned empty tensors
        # and custom initialization is silently skipped)
        import deepspeed

        from transformers import AutoConfig, AutoModelForCausalLM

        ds_config = self._get_zero3_ds_config(bf16={"enabled": True})

        config = AutoConfig.from_pretrained(GPT2_TINY)

        # 1. Baseline: from_config without DeepSpeed, in bf16 to match ZeRO-3 dtype
        torch.manual_seed(42)
        baseline_model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
        baseline_std = baseline_model.transformer.h[0].attn.c_attn.weight.data.float().std().item()
        del baseline_model

        # 2. ZeRO-3: from_config with DeepSpeed
        torch.manual_seed(42)
        HfDeepSpeedConfig(ds_config)

        with mockenv_context(**self.dist_env_1_gpu):
            model = AutoModelForCausalLM.from_config(config)

        param = model.transformer.h[0].attn.c_attn.weight
        with deepspeed.zero.GatheredParameters([param]):
            zero3_std = param.data.float().std().item()

        # Weight std should be in the same ballpark between baseline and ZeRO-3.
        ratio = zero3_std / baseline_std
        self.assertAlmostEqual(
            ratio,
            1.0,
            delta=0.3,
            msg=(
                f"ZeRO-3 from_config() weight init diverges from baseline: "
                f"baseline_std={baseline_std:.6f}, zero3_std={zero3_std:.6f}, ratio={ratio:.4f}"
            ),
        )

    def test_init_zero3_missing_params(self):
        # test that zero.Init() for missing parameters works correctly under zero3
        import deepspeed

        from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2PreTrainedModel

        class TinyGPT2WithUninitializedWeights(GPT2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.transformer = GPT2Model(config)
                self.new_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=True)

                # Initialize weights and apply final processing
                self.post_init()

            def forward(self, *args, **kwargs):
                transformer_outputs = self.transformer(*args, **kwargs)
                hidden_states = transformer_outputs[0]
                return self.new_head(hidden_states).float()

            def _init_weights(self, module):
                super()._init_weights(module)
                if module is self.new_head:
                    nn.init.constant_(self.new_head.weight.data, -100.0)
                    nn.init.constant_(self.new_head.bias.data, 100.0)

        ds_config = self._get_zero3_ds_config()

        # With zero3
        dschf = HfDeepSpeedConfig(ds_config)  # noqa: F841 — prevent GC of weak-ref config
        model, cl = self._load_with_logging(TinyGPT2WithUninitializedWeights, GPT2_TINY, expect_zero3=True)
        self.assertRegex(cl.out, r"new_head\.(weight|bias)\s*\|\s*MISSING")
        with deepspeed.zero.GatheredParameters([model.new_head.weight, model.new_head.bias]):
            self.assertTrue(
                torch.allclose(model.new_head.weight, torch.tensor(-100.0, device=model.new_head.weight.device))
            )
            self.assertTrue(
                torch.allclose(model.new_head.bias, torch.tensor(+100.0, device=model.new_head.bias.device))
            )

        # Without zero3
        del ds_config["zero_optimization"]
        dschf = HfDeepSpeedConfig(ds_config)  # noqa: F841
        model, cl = self._load_with_logging(TinyGPT2WithUninitializedWeights, GPT2_TINY, expect_zero3=False)
        self.assertRegex(cl.out, r"new_head\.(weight|bias)\s*\|\s*MISSING")
        self.assertTrue(
            torch.allclose(model.new_head.weight, torch.tensor(-100.0, device=model.new_head.weight.device))
        )
        self.assertTrue(torch.allclose(model.new_head.bias, torch.tensor(+100.0, device=model.new_head.bias.device)))

    def test_arange_bf16(self):
        # Tests that configuring DeepSpeed with 16 bits does not cause float `torch.arange()` tensors to be cast down.
        # NOTE -- this assumes that the function calls have the following downcast-preventing pattern, i.e.
        # `torch.arange(...,dtype=torch.int64)` followed by a cast like `.to(torch.float32)`. If this pattern is
        # NOT applied (e.g. `torch.arange(...,dtype=torch.float32)` is used), DeepSpeed can automatically cast it down
        # at init time. See https://github.com/huggingface/transformers/issues/28685 for more info.

        ds_config = self._get_zero3_ds_config(bf16={"enabled": True})

        dschf = HfDeepSpeedConfig(ds_config)
        self.assertTrue(dschf.is_zero3())
        self.assertTrue(is_deepspeed_zero3_enabled())

        model, _ = self._load_with_logging(AutoModel, GPTJ_TINY, expect_zero3=True, dtype=torch.float32)

        # The model weights are in BF16 as per deepspeed config
        self.assertTrue(str(model.h[0].attn.q_proj.weight.dtype) == "torch.bfloat16")
        good_deepspeed_sin_cos = model.h[0].attn.embed_positions

        good_deepspeed_create_sinusoidal_positions = transformers.models.gptj.modeling_gptj.create_sinusoidal_positions

        good_torch_sin_cos = good_deepspeed_create_sinusoidal_positions(
            model.config.max_position_embeddings, model.config.rotary_dim
        )
        # check that we get the same results either with torch or deepspeed
        torch.testing.assert_close(good_torch_sin_cos, good_deepspeed_sin_cos.cpu())

    def test_init_zero3_moe_weight_conversion(self):
        # test that weight conversions (MoE expert fusion) work correctly under zero3
        import tempfile

        import deepspeed

        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

        tiny_config = Qwen3MoeConfig(
            vocab_size=99,
            hidden_size=32,
            intermediate_size=32,
            moe_intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_experts=8,
            num_experts_per_tok=2,
        )

        ds_config = self._get_zero3_ds_config()

        dschf = HfDeepSpeedConfig(ds_config)
        self.assertTrue(dschf.is_zero3())
        self.assertTrue(is_deepspeed_zero3_enabled())

        with tempfile.TemporaryDirectory() as tmpdirname:
            with LoggingLevel(logging.INFO):
                with mockenv_context(**self.dist_env_1_gpu):
                    model = Qwen3MoeForCausalLM(tiny_config)
                    reference_weights = {k: v.clone() for k, v in model.state_dict().items()}
                    model.save_pretrained(tmpdirname)

            (loaded_model, loading_info), _ = self._load_with_logging(
                Qwen3MoeForCausalLM, tmpdirname, expect_zero3=True, output_loading_info=True
            )
            self.assertEqual(len(loading_info["missing_keys"]), 0, f"Missing keys: {loading_info['missing_keys']}")
            self.assertEqual(
                len(loading_info["unexpected_keys"]), 0, f"Unexpected keys: {loading_info['unexpected_keys']}"
            )

            # gather all params and verify they match the original weights exactly
            all_params = list(loaded_model.named_parameters())
            with deepspeed.zero.GatheredParameters([p for _, p in all_params], modifier_rank=0):
                for name, param in all_params:
                    torch.testing.assert_close(
                        param.data.cpu(),
                        reference_weights[name].cpu(),
                        msg=f"Parameter '{name}' doesn't match reference weights",
                    )

    def test_init_zero3_variance_scaling(self):
        """
        Tests whether variance scaling initializations (`lecun_normal_`, `default_flax_embed_init_`) work correctly
        with DeepSpeed ZeRO-3, e.g. as in SigLIP models. It indirectly checks for the `_is_hf_initialized` flag to
        prevent re-initialization in ZeRO-3 environments. See #43574
        """
        import tempfile

        from transformers import (
            SiglipConfig,
            SiglipModel,
            SiglipTextConfig,
            SiglipVisionConfig,
        )

        text_cfg = SiglipTextConfig(
            vocab_size=64,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=16,
        )

        vision_cfg = SiglipVisionConfig(
            image_size=4,
            patch_size=2,
            num_channels=3,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )

        cfg = SiglipConfig(text_config=text_cfg.to_dict(), vision_config=vision_cfg.to_dict())

        with tempfile.TemporaryDirectory() as tmpdirname:
            model = SiglipModel(cfg).eval()
            model.save_pretrained(tmpdirname)

            ds_config = self._get_zero3_ds_config()
            dschf = HfDeepSpeedConfig(ds_config)
            self.assertTrue(dschf.is_zero3())
            self.assertTrue(is_deepspeed_zero3_enabled())

            model, _ = self._load_with_logging(SiglipModel, tmpdirname, expect_zero3=True)

        self.assertIsNotNone(model)

    def test_resize_token_embeddings_zero3(self):
        """resize_token_embeddings requires param gathering outside of forward under ZeRO-3."""
        import deepspeed

        from transformers import AutoModelForCausalLM

        ds_config = self._get_zero3_ds_config(bf16={"enabled": True})
        HfDeepSpeedConfig(ds_config)

        with mockenv_context(**self.dist_env_1_gpu):
            model = AutoModelForCausalLM.from_pretrained(GPT2_TINY)

        original_size = model.get_input_embeddings().weight.shape[0]
        new_size = original_size + 10
        model.resize_token_embeddings(new_size)

        embedding = model.get_input_embeddings()
        with deepspeed.zero.GatheredParameters([embedding.weight]):
            self.assertEqual(embedding.weight.shape[0], new_size)


# ---------------------------------------------------------------------------
# Model Zoo — test many architectures with DeepSpeed + zero_to_fp32 recovery
# ---------------------------------------------------------------------------

_ZOO_MODELS = {
    "albert": "hf-internal-testing/tiny-albert",
    "bart": "sshleifer/bart-tiny-random",
    "bert": "hf-internal-testing/tiny-bert",
    "bigbird_pegasus": "hf-internal-testing/tiny-random-bigbird_pegasus",
    "blenderbot": "hf-internal-testing/tiny-random-blenderbot",
    "bloom": "bigscience/bigscience-small-testing",
    "deberta": "hf-internal-testing/tiny-random-deberta",
    "deberta-v2": "hf-internal-testing/tiny-random-deberta-v2",
    "distilbert": "sshleifer/tiny-distilbert-base-cased",
    "electra": "hf-internal-testing/tiny-electra",
    "funnel": "hf-internal-testing/tiny-random-funnel",
    "gpt2": GPT2_TINY,
    "gpt_neo": "hf-internal-testing/tiny-random-gpt_neo",
    "gptj": GPTJ_TINY,
    "layoutlm": "hf-internal-testing/tiny-layoutlm",
    "led": "hf-internal-testing/tiny-random-led",
    "longformer": "hf-internal-testing/tiny-random-longformer",
    "m2m_100": "stas/tiny-m2m_100",
    "mobilebert": "hf-internal-testing/tiny-random-mobilebert",
    "mpnet": "hf-internal-testing/tiny-random-mpnet",
    "prophetnet": "hf-internal-testing/tiny-random-prophetnet",
    "roberta": "sshleifer/tiny-distilroberta-base",
    "squeezebert": "hf-internal-testing/tiny-random-squeezebert",
    "t5": T5_TINY,
    "t5_v1": "hf-internal-testing/tiny-random-t5-v1.1",
    "vit": "hf-internal-testing/tiny-random-vit",
    "xlm-roberta": "hf-internal-testing/tiny-xlm-roberta",
    "xlnet": "sshleifer/tiny-xlnet-base-cased",
}

_ZOO_FIXTURE_DIR = get_tests_dir("fixtures")
_ZOO_SAMPLES_DIR = f"{_ZOO_FIXTURE_DIR}/tests_samples"
_ZOO_SCRIPTS_DIR = f"{os.path.join(os.path.dirname(get_tests_dir()))}/examples/pytorch"
_ZOO_VIT_FEATURE_EXTRACTOR = os.path.join(SCRIPTS_DIR, "vit_feature_extractor.json")


def _make_zoo_tasks():
    """Build {task_model: (script, script_args)} for each task/model combo."""
    tasks2models = {
        "trans": ["bart", "m2m_100", "t5", "t5_v1"],
        "clm": ["bigbird_pegasus", "blenderbot", "bloom", "gpt2", "gpt_neo", "gptj", "xlm-roberta", "prophetnet"],
        "mlm": ["albert", "deberta", "deberta-v2", "distilbert", "electra", "layoutlm"],
        "qa": ["led", "longformer", "mobilebert", "mpnet", "roberta", "squeezebert"],
        "clas": ["bert", "xlnet"],
        "img_clas": ["vit"],
    }

    # task -> (script_path, task-specific args)
    task_defs = {
        "trans": (
            f"{_ZOO_SCRIPTS_DIR}/translation/run_translation.py",
            f"--train_file {_ZOO_SAMPLES_DIR}/wmt_en_ro/train.json --source_lang en --target_lang ro "
            f"--max_source_length 12 --max_target_length 12".split(),
        ),
        "clm": (
            f"{_ZOO_SCRIPTS_DIR}/language-modeling/run_clm.py",
            f"--train_file {_ZOO_FIXTURE_DIR}/sample_text.txt --block_size 8".split(),
        ),
        "mlm": (
            f"{_ZOO_SCRIPTS_DIR}/language-modeling/run_mlm.py",
            f"--train_file {_ZOO_FIXTURE_DIR}/sample_text.txt".split(),
        ),
        "qa": (
            f"{_ZOO_SCRIPTS_DIR}/question-answering/run_qa.py",
            f"--train_file {_ZOO_SAMPLES_DIR}/SQUAD/sample.json".split(),
        ),
        "clas": (
            f"{_ZOO_SCRIPTS_DIR}/text-classification/run_glue.py",
            f"--train_file {_ZOO_SAMPLES_DIR}/MRPC/train.csv --max_seq_length 12 --task_name MRPC".split(),
        ),
        "img_clas": (
            f"{_ZOO_SCRIPTS_DIR}/image-classification/run_image_classification.py",
            f"--dataset_name hf-internal-testing/cats_vs_dogs_sample --remove_unused_columns False "
            f"--max_steps 10 --image_processor_name {_ZOO_VIT_FEATURE_EXTRACTOR} "
            f"--label_column_name labels".split(),
        ),
    }

    common_args = "--do_train --max_train_samples 4 --per_device_train_batch_size 2 --num_train_epochs 1 --fp16 --save_steps 1".split()

    result = {}
    for task, models in tasks2models.items():
        script, task_args = task_defs[task]
        for model in models:
            model_args = ["--model_name_or_path", _ZOO_MODELS[model]]
            result[f"{task}_{model}"] = (script, task_args + model_args + common_args)

    return result


_zoo_tasks = _make_zoo_tasks()
_zoo_params = list(itertools.product(stages, _zoo_tasks.keys()))


@slow
@require_deepspeed
@require_torch_accelerator
class TestDeepSpeedModelZoo(DeepSpeedCommandsMixin, TestCasePlus):
    """Test many model architectures with DeepSpeed (fp16 mixed precision) via example scripts + zero_to_fp32 recovery."""

    @parameterized.expand(_zoo_params, name_func=_parameterized_custom_name_func)
    def test_zero_to_fp32(self, stage, task):
        script, script_args = _zoo_tasks[task]
        output_dir = self.get_auto_remove_tmp_dir()

        # 1. Train and save a checkpoint
        cmd = self.get_accelerate_cmd(
            script,
            config_file=DS_CONFIGS[stage],
            script_args=script_args + ["--output_dir", output_dir],
        )
        execute_subprocess_async(cmd, env=self.get_env())

        # 2. Recover FP32 weights from the ZeRO checkpoint
        chkpt_dir = f"{output_dir}/checkpoint-1"
        recovered_model_path = f"{chkpt_dir}/out.bin"
        subprocess.check_call(f"{chkpt_dir}/zero_to_fp32.py {chkpt_dir} {recovered_model_path}", shell=True)
        assert os.path.exists(recovered_model_path), f"{recovered_model_path} was not found"
