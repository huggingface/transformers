# Copyright 2024 The HuggingFace Team. All rights reserved.
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
FSDP-specific distributed trainer tests.
"""

import itertools
import json
import os
import textwrap
from functools import partial
from pathlib import Path

from parameterized import parameterized

from tests.trainer.trainer_test_utils import TrainerIntegrationCommon, get_regression_trainer  # noqa
from transformers import is_torch_available
from transformers.testing_utils import (
    TestCasePlus,
    backend_device_count,
    execute_subprocess_async,
    get_torch_dist_unique_port,
    mockenv_context,
    require_accelerate,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    run_first,
    slow,
    torch_device,
    torchrun,
)
from transformers.trainer_utils import FSDPOption, set_seed
from transformers.utils import (
    is_accelerate_available,
    is_torch_bf16_available_on_device,
    is_torch_fp16_available_on_device,
)

from .test_trainer_distributed import CONFIGS_DIR, SCRIPTS_DIR, TRAIN_SCRIPT, TrainerDistributedCommon


if is_torch_available():
    import torch

    from transformers.trainer import FSDP_MODEL_NAME

# Base accelerate configs (version only — model-specific settings via launch args)
FSDP_CONFIG_FILE = os.path.join(CONFIGS_DIR, "fsdp.yaml")
FSDP2_CONFIG_FILE = os.path.join(CONFIGS_DIR, "fsdp2.yaml")
FSDP2_CP_CONFIG_FILE = os.path.join(CONFIGS_DIR, "fsdp2_cp.yaml")
FSDP_GENERATE_SCRIPT = os.path.join(SCRIPTS_DIR, "fsdp_generate.py")

FSDP_CONFIGS = {
    "fsdp1": FSDP_CONFIG_FILE,
    "fsdp2": FSDP2_CONFIG_FILE,
}

# Launch args shared by all training tests
TRAIN_LAUNCH_ARGS = [
    "--fsdp_auto_wrap_policy",
    "TRANSFORMER_BASED_WRAP",
]

dtypes = []
if is_torch_bf16_available_on_device(torch_device):
    dtypes += ["bf16"]
if is_torch_fp16_available_on_device(torch_device):
    dtypes += ["fp16"]

sharding_strategies = ["full_shard", "shard_grad_op"]  # zero3 and zero2
fsdp_versions = ["fsdp1", "fsdp2"]

config_params = list(itertools.product(sharding_strategies, dtypes))
# Mixed precision: model loaded in fp32, training with --bf16/--fp16
mixed_precision_params = list(itertools.product(sharding_strategies, dtypes, fsdp_versions))
# Pure dtype: model loaded in target dtype, no mixed precision flags
pure_dtype_params = list(itertools.product(["fp32"] + dtypes, fsdp_versions))

resume_params = [
    ("FULL_STATE_DICT", "fsdp1"),  # FULL_STATE_DICT only supported for fsdp1
    ("SHARDED_STATE_DICT", "fsdp1"),
    ("SHARDED_STATE_DICT", "fsdp2"),
]

set_seed(42)


if is_torch_available():
    # hack to restore original logging level pre #21700
    get_regression_trainer = partial(get_regression_trainer, log_level="info")

if is_accelerate_available():
    from accelerate.utils.constants import FSDP_SHARDING_STRATEGY


def _parameterized_custom_name_func(func, param_num, param):
    # customize the test name generator function as we want both params to appear in the sub-test
    # name, as by default it shows only the first param
    param_based_name = parameterized.to_safe_name("_".join(str(x) for x in param.args))
    return f"{func.__name__}_{param_based_name}"


# ---------------------------------------------------------------------------
# Command mixins
# ---------------------------------------------------------------------------


class FSDPCommandsMixin:
    """Provides ``get_torchrun_cmd`` and ``get_accelerate_cmd`` for FSDP."""

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
# Config parsing tests (no distributed training runs)
# ---------------------------------------------------------------------------


@require_accelerate
@require_torch_accelerator
class TestFSDPConfig(TestCasePlus):
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
        self.accelerate_fsdp_config = {
            "fsdp_activation_checkpointing": False,
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_backward_prefetch": "BACKWARD_PRE",
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_forward_prefetch": False,
            "fsdp_offload_params": False,
            "fsdp_reshard_after_forward": "FULL_SHARD",
            "fsdp_state_dict_type": "FULL_STATE_DICT",
            "fsdp_sync_module_states": True,
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "fsdp_use_orig_params": True,
            "fsdp_version": 1,
        }

        self.fsdp_config = {
            "backward_prefetch": "BACKWARD_PRE",
            "forward_prefetch": "false",
            "limit_all_gathers": "false",
            "use_orig_params": "true",
            "sync_module_states": "true",
            "cpu_ram_efficient_loading": "true",
            "activation_checkpointing": "false",
            "min_num_params": 1,
        }

    @parameterized.expand(config_params, name_func=_parameterized_custom_name_func)
    def test_accelerate_fsdp_config(self, sharding_strategy, dtype):
        output_dir = self.get_auto_remove_tmp_dir()
        kwargs = {
            "output_dir": output_dir,
            "train_len": 128,
            "save_steps": 5,
            "learning_rate": 0.1,
            "fsdp": f"{sharding_strategy} offload auto_wrap",
            "fsdp_config": self.accelerate_fsdp_config,
        }
        kwargs[dtype] = True
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(**kwargs)
            self.assertEqual(trainer.args.fsdp[0], sharding_strategy)
            self.assertEqual(trainer.args.fsdp[1], FSDPOption.OFFLOAD)
            self.assertEqual(trainer.args.fsdp[2], FSDPOption.AUTO_WRAP)
            for k, v in trainer.args.fsdp_config.items():
                self.assertTrue(k in self.accelerate_fsdp_config)
                self.assertEqual(v, self.accelerate_fsdp_config[k])

    def test_torchrun_fsdp_config(self):
        """Verify that --fsdp + --fsdp_config (torchrun-style) are parsed correctly."""
        output_dir = self.get_auto_remove_tmp_dir()
        fsdp_config = {"fsdp_transformer_layer_cls_to_wrap": "Qwen2DecoderLayer"}
        kwargs = {
            "output_dir": output_dir,
            "train_len": 128,
            "save_steps": 5,
            "learning_rate": 0.1,
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": fsdp_config,
            "bf16": True,
        }
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(**kwargs)
            self.assertEqual(trainer.args.fsdp[0], "full_shard")
            self.assertEqual(trainer.args.fsdp[1], FSDPOption.AUTO_WRAP)
            # fsdp_ prefix is stripped and value is normalized to a list during parsing
            self.assertIn("Qwen2DecoderLayer", trainer.args.fsdp_config["transformer_layer_cls_to_wrap"])

    @parameterized.expand(config_params, name_func=_parameterized_custom_name_func)
    def test_fsdp_config(self, sharding_strategy, dtype):
        output_dir = self.get_auto_remove_tmp_dir()
        kwargs = {
            "output_dir": output_dir,
            "train_len": 128,
            "save_steps": 5,
            "learning_rate": 0.1,
            "fsdp": f"{sharding_strategy} offload auto_wrap",
            "fsdp_config": self.fsdp_config,
        }
        kwargs[dtype] = True
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(**kwargs)
            self.assertEqual(trainer.args.fsdp[0], sharding_strategy)
            self.assertEqual(trainer.args.fsdp[1], FSDPOption.OFFLOAD)
            self.assertEqual(trainer.args.fsdp[2], FSDPOption.AUTO_WRAP)
            for k, v in trainer.args.fsdp_config.items():
                self.assertEqual(v, self.fsdp_config[k])


# ---------------------------------------------------------------------------
# FSDP distributed tests
# ---------------------------------------------------------------------------


class TestTrainerDistributedFSDP(FSDPCommandsMixin, TestCasePlus):
    def _run_env_check(self, cmd, num_processes):
        """Run the env check script and return per-rank results."""
        execute_subprocess_async(cmd, env=self.get_env())
        # output_dir is always the last script_arg value
        output_dir = cmd[cmd.index("--output_dir") + 1]
        results = []
        for rank in range(num_processes):
            with open(os.path.join(output_dir, f"env_rank{rank}.json")) as f:
                results.append(json.load(f))
        return results

    @run_first
    @require_accelerate
    @require_torch_multi_accelerator
    def test_torchrun_accelerate_fsdp1_env_parity(self):
        """Verify torchrun+--fsdp and accelerate launch produce the same FSDP1 env."""
        script = os.path.join(SCRIPTS_DIR, "torchrun_env_check.py")
        num_processes = backend_device_count(torch_device)

        torchrun_dir = self.get_auto_remove_tmp_dir()
        torchrun_results = self._run_env_check(
            self.get_torchrun_cmd(
                script,
                script_args=[
                    "--output_dir",
                    torchrun_dir,
                    "--fsdp",
                    "full_shard",
                    "--fsdp_config",
                    '{"fsdp_version": 1}',
                ],
                num_processes=num_processes,
            ),
            num_processes,
        )

        accel_dir = self.get_auto_remove_tmp_dir()
        accel_results = self._run_env_check(
            self.get_accelerate_cmd(
                script, FSDP_CONFIG_FILE, script_args=["--output_dir", accel_dir], num_processes=num_processes
            ),
            num_processes,
        )

        self._check_parity(torchrun_results, accel_results, num_processes, expected_fsdp_version=1)

    @run_first
    @require_accelerate
    @require_torch_multi_accelerator
    def test_torchrun_accelerate_fsdp2_env_parity(self):
        """Verify torchrun+--fsdp and accelerate launch produce the same FSDP2 env."""
        script = os.path.join(SCRIPTS_DIR, "torchrun_env_check.py")
        num_processes = backend_device_count(torch_device)

        torchrun_dir = self.get_auto_remove_tmp_dir()
        torchrun_results = self._run_env_check(
            self.get_torchrun_cmd(
                script,
                script_args=[
                    "--output_dir",
                    torchrun_dir,
                    "--fsdp",
                    "full_shard",
                    "--fsdp_config",
                    '{"fsdp_version": 2}',
                ],
                num_processes=num_processes,
            ),
            num_processes,
        )

        accel_dir = self.get_auto_remove_tmp_dir()
        accel_results = self._run_env_check(
            self.get_accelerate_cmd(
                script, FSDP2_CONFIG_FILE, script_args=["--output_dir", accel_dir], num_processes=num_processes
            ),
            num_processes,
        )

        self._check_parity(torchrun_results, accel_results, num_processes, expected_fsdp_version=2)

    def _check_parity(self, torchrun_results, accel_results, num_processes, expected_fsdp_version):
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

                # Both should have FSDP enabled with the correct version
                self.assertEqual(info["accelerator_distributed_type"], "DistributedType.FSDP")
                self.assertTrue(info["trainer_is_fsdp_enabled"])
                self.assertFalse(info["trainer_is_deepspeed_enabled"])
                self.assertEqual(info["fsdp_version"], expected_fsdp_version)
                self.assertNotIn("deepspeed_zero_stage", info)


# ---------------------------------------------------------------------------
# All distributed FSDP training tests
# ---------------------------------------------------------------------------
@slow
@run_first
@require_accelerate
@require_torch_multi_accelerator
class TestTrainerDistributedFSDPCommon(
    FSDPCommandsMixin, TrainerDistributedCommon, TestCasePlus, TrainerIntegrationCommon
):
    # -------------------------------------------------------------------
    # FSDP training — accelerate (parameterized over fsdp version)
    # -------------------------------------------------------------------

    # Pure dtype training: model loaded in target dtype, no mixed precision
    @parameterized.expand(pure_dtype_params, name_func=_parameterized_custom_name_func)
    def test_training(self, dtype, fsdp_version):
        self.check_training(dtype, config_file=FSDP_CONFIGS[fsdp_version])

    # Mixed precision: model loaded in fp32, training with --bf16/--fp16
    @parameterized.expand(mixed_precision_params, name_func=_parameterized_custom_name_func)
    def test_training_mixed_precision(self, sharding_strategy, dtype, fsdp_version):
        sharding_idx = FSDP_SHARDING_STRATEGY.index(sharding_strategy.upper()) + 1
        launch_args = list(TRAIN_LAUNCH_ARGS) + ["--fsdp_sharding_strategy", str(sharding_idx)]
        self.check_mixed_precision(dtype, config_file=FSDP_CONFIGS[fsdp_version], launch_args=launch_args)

    @parameterized.expand(["true", "false"], name_func=_parameterized_custom_name_func)
    def test_fsdp2_cpu_ram_efficient_loading(self, cpu_ram_efficient_loading):
        launch_args = list(TRAIN_LAUNCH_ARGS) + [
            "--fsdp_cpu_ram_efficient_loading",
            cpu_ram_efficient_loading,
        ]
        self.check_training("bf16", config_file=FSDP2_CONFIG_FILE, launch_args=launch_args)

    @parameterized.expand(fsdp_versions, name_func=_parameterized_custom_name_func)
    def test_training_with_gradient_accumulation(self, fsdp_version):
        self.check_gradient_accumulation(config_file=FSDP_CONFIGS[fsdp_version])

    @parameterized.expand(fsdp_versions, name_func=_parameterized_custom_name_func)
    def test_basic_run_with_cpu_offload(self, fsdp_version):
        output_dir = self.get_auto_remove_tmp_dir()
        args = self._get_default_script_args(output_dir) + ["--bf16", "--max_steps", "10"]
        launch_args = list(TRAIN_LAUNCH_ARGS) + ["--fsdp_offload_params", "true"]
        execute_subprocess_async(
            self.get_accelerate_cmd(
                TRAIN_SCRIPT, script_args=args, config_file=FSDP_CONFIGS[fsdp_version], launch_args=launch_args
            ),
            env=self.get_env(),
        )

    @parameterized.expand(resume_params, name_func=_parameterized_custom_name_func)
    def test_training_and_can_resume_normally(self, state_dict_type, fsdp_version):
        output_dir = self.get_auto_remove_tmp_dir()
        args = self._get_default_script_args(output_dir, num_epochs=2, logging_steps=2, save_steps=2)

        launch_args = list(TRAIN_LAUNCH_ARGS) + ["--fsdp_state_dict_type", state_dict_type]
        cmd_kwargs = {"config_file": FSDP_CONFIGS[fsdp_version], "launch_args": launch_args}

        logs = self._train_and_get_log_history(
            self.get_accelerate_cmd(TRAIN_SCRIPT, script_args=args, **cmd_kwargs),
            output_dir,
        )

        # resume from ckpt
        checkpoint = os.path.join(output_dir, "checkpoint-2")
        resume_args = args + ["--resume_from_checkpoint", checkpoint]

        is_fsdp_ckpt = os.path.isdir(checkpoint) and (
            # this checks the FSDP state dict when `SHARDED_STATE_DICT` is used
            any(
                FSDP_MODEL_NAME in folder_name
                for folder_name in os.listdir(checkpoint)
                if os.path.isdir(os.path.join(checkpoint, folder_name))
            )
            # this checks the FSDP state dict when `FULL_STATE_DICT` is used
            or os.path.isfile(os.path.join(checkpoint, f"{FSDP_MODEL_NAME}.bin"))
        )
        self.assertTrue(is_fsdp_ckpt)

        logs_resume = self._train_and_get_log_history(
            self.get_accelerate_cmd(TRAIN_SCRIPT, script_args=resume_args, **cmd_kwargs),
            output_dir,
        )

        for log, log1 in zip(logs, logs_resume):
            if "learning_rate" in log:
                self.assertAlmostEqual(log["learning_rate"], log1["learning_rate"], delta=1e-5)

    # -------------------------------------------------------------------
    # Context parallel tests
    # -------------------------------------------------------------------
    def test_cp_equivalence(self):
        """Test that CP produces the same losses as without CP."""

        # CP doesn't work with Qwen2 (DTensor mixing error), so we use Llama here.
        launch_args = list(TRAIN_LAUNCH_ARGS) + ["--fsdp_state_dict_type", "SHARDED_STATE_DICT"]
        cp_script_args = [
            "--model_name",
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "--max_steps",
            "10",
            "--per_device_train_batch_size",
            "1",
            "--seed",
            "42",
            "--logging_steps",
            "1",
            "--save_strategy",
            "no",
            "--model_dtype",
            "fp32",
            "--attn_implementation",
            "sdpa",
            "--pad_to_multiple_of",
            "4",
        ]

        # Step 1: Run with CP enabled (cp_size=2)
        cp_yes_output_dir = Path(self.get_auto_remove_tmp_dir()).resolve()
        cp_yes_losses_path = cp_yes_output_dir / "cp_yes_losses.json"
        cmd = self.get_accelerate_cmd(
            TRAIN_SCRIPT,
            config_file=FSDP2_CP_CONFIG_FILE,
            launch_args=launch_args,
            script_args=["--output_dir", str(cp_yes_output_dir), "--loss_output_file", str(cp_yes_losses_path)]
            + cp_script_args,
        )
        execute_subprocess_async(cmd, env=self.get_env())

        # Step 2: Run without CP (FSDP with num_processes=1, no parallelism_config)
        cp_no_output_dir = Path(self.get_auto_remove_tmp_dir()).resolve()
        cp_no_losses_path = cp_no_output_dir / "cp_no_losses.json"

        cmd = self.get_accelerate_cmd(
            TRAIN_SCRIPT,
            config_file=FSDP2_CONFIG_FILE,
            launch_args=launch_args,
            script_args=[
                "--output_dir",
                str(cp_no_output_dir),
                "--loss_output_file",
                str(cp_no_losses_path),
            ]
            + cp_script_args,
            num_processes=1,
        )
        execute_subprocess_async(cmd, env=self.get_env())

        # Compare losses
        with open(cp_yes_losses_path) as f:
            cp_yes_losses = json.load(f)
        with open(cp_no_losses_path) as f:
            cp_no_losses = json.load(f)

        assert len(cp_yes_losses) == len(cp_no_losses), (
            f"Different number of losses: CP has {len(cp_yes_losses)}, no-CP has {len(cp_no_losses)}"
        )

        cp_yes_losses_tensor = torch.tensor(cp_yes_losses)
        cp_no_losses_tensor = torch.tensor(cp_no_losses)

        torch.testing.assert_close(
            cp_yes_losses_tensor,
            cp_no_losses_tensor,
            rtol=2e-2,
            atol=2e-2,
            msg=f"CP losses {cp_yes_losses} do not match non-CP losses {cp_no_losses}",
        )

    # -------------------------------------------------------------------
    # FSDP eval tests
    # -------------------------------------------------------------------
    def test_eval(self):
        self.check_eval(config_file=FSDP_CONFIG_FILE)

    # -------------------------------------------------------------------
    # FSDP generation tests (moved from tests/generation/test_fsdp.py)
    # -------------------------------------------------------------------
    def test_fsdp_generate(self):
        cmd = self.get_accelerate_cmd(
            FSDP_GENERATE_SCRIPT,
            config_file=FSDP_CONFIG_FILE,
            script_args=["--fsdp"],
        )
        execute_subprocess_async(cmd, env=self.get_env())

    def test_fsdp2_generate(self):
        cmd = self.get_accelerate_cmd(
            FSDP_GENERATE_SCRIPT,
            config_file=FSDP2_CONFIG_FILE,
            script_args=["--fsdp2"],
        )
        execute_subprocess_async(cmd, env=self.get_env())


# ---------------------------------------------------------------------------
# FSDP generic task model sharding (moved from tests/generation/test_fsdp.py)
# ---------------------------------------------------------------------------


@require_torch_multi_accelerator
class TestFSDPGenericTaskModel(TestCasePlus):
    nproc_per_node = 2

    def test_generic_task_model_can_be_sharded(self):
        script_to_run = textwrap.dedent(
            """
            import torch
            from torch.distributed.fsdp import fully_shard
            from transformers import AutoModelForTokenClassification

            current_accelerator = torch.accelerator.current_accelerator(check_available=True)
            accelerator_type = "cpu" if current_accelerator is None else current_accelerator.type
            torch_accelerator_module = getattr(torch, accelerator_type, torch.cuda)

            backend = "gloo"
            if accelerator_type == "cuda":
                backend = "nccl"
            elif accelerator_type == "xpu":
                backend = "xccl"

            torch.distributed.init_process_group(
                backend=backend, init_method="env://"
            )
            rank = torch.distributed.get_rank()
            if torch_accelerator_module.is_available():
                torch_accelerator_module.set_device(rank)

            # Make sure it works
            model = AutoModelForTokenClassification.from_pretrained("Qwen/Qwen2-0.5B")
            module = fully_shard(model)

            torch.distributed.destroy_process_group()
            """
        )
        torchrun(script_to_run, self.nproc_per_node, env=self.get_env())
