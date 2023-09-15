# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import itertools
import json
import os
import unittest
from copy import deepcopy
from functools import partial

import datasets
from parameterized import parameterized

import tests.trainer.test_trainer
from tests.trainer.test_trainer import TrainerIntegrationCommon  # noqa
from transformers import AutoModel, TrainingArguments, is_torch_available, logging
from transformers.trainer_callback import TrainerState

from transformers.testing_utils import (
    CaptureLogger,
    CaptureStd,
    CaptureStderr,
    LoggingLevel,
    TestCasePlus,
    execute_subprocess_async,
    get_gpu_count,
    mockenv_context,
    require_deepspeed,
    require_optuna,
    require_torch_gpu,
    require_accelerate,
    require_fsdp,
    require_torch_multi_gpu,
    slow,
)
from transformers.trainer_utils import get_last_checkpoint, set_seed, FSDPOption
from transformers.utils import WEIGHTS_NAME, is_torch_bf16_gpu_available, is_accelerate_available


# default torch.distributed port
DEFAULT_MASTER_PORT = "10999"
set_seed(42)


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


if is_torch_available():
    from tests.trainer.test_trainer import (  # noqa
        RegressionModelConfig,
        RegressionPreTrainedModel,
    )

    # hack to restore original logging level pre #21700
    get_regression_trainer = partial(tests.trainer.test_trainer.get_regression_trainer, log_level="info")

if is_accelerate_available():
    from accelerate.utils.constants import (
        FSDP_AUTO_WRAP_POLICY,
        FSDP_BACKWARD_PREFETCH,
        FSDP_SHARDING_STRATEGY,
        FSDP_STATE_DICT_TYPE,
        FSDP_PYTORCH_VERSION,
    )

    require_fsdp_version = partial(require_fsdp, min_version=FSDP_PYTORCH_VERSION)


def get_launcher(distributed=False, use_accelerate=False):
    # 1. explicitly set --num_nodes=1 just in case these tests end up run on a multi-node setup
    # - it won't be able to handle that
    # 2. for now testing with just 2 gpus max (since some quality tests may give different
    # results with mode gpus because we use very little data)
    num_gpus = min(2, get_gpu_count()) if distributed else 1
    master_port = get_master_port(real_launcher=True)
    if use_accelerate:
        return f"""accelerate launch
        --num_processes {num_gpus}
        --main_process_port {master_port}
        --use_fsdp
        --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP
        --fsdp_state_dict_type SHARDED_STATE_DICT
        --fsdp_transformer_layer_cls_to_wrap BertLayer""".split()
    return f"torchrun --nnodes 1 --nproc-per-node {num_gpus} --master-port {master_port}".split()


FP16 = "fp16"
BF16 = "bf16"
if is_torch_bf16_gpu_available():
    dtypes = [FP16, BF16]
else:
    dtypes = [FP16]

FULL_SHARD = "full_shard"
SHARD_GRAD_OP = "shard_grad_op"
sharding_strategies = [FULL_SHARD, SHARD_GRAD_OP]


def parameterized_custom_name_func(func, param_num, param):
    # customize the test name generator function as we want both params to appear in the sub-test
    # name, as by default it shows only the first param
    param_based_name = parameterized.to_safe_name("_".join(str(x) for x in param.args))
    return f"{func.__name__}_{param_based_name}"


params = list(itertools.product(sharding_strategies, dtypes))
params_with_state_dict_type = list(itertools.product(sharding_strategies, dtypes, FSDP_STATE_DICT_TYPE))


@require_accelerate
@require_torch_gpu
@require_fsdp_version
class TrainerIntegrationFSDP(TestCasePlus, TrainerIntegrationCommon):
    def setUp(self):
        super().setUp()
        master_port = get_master_port(real_launcher=False)
        self.dist_env_1_gpu = {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": master_port,
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        }

        self.fsdp_config = {
            "backward_prefetch": "backward_pre",
            "forward_prefetch": "False",
            "limit_all_gathers": "False",
            "use_orig_params": "True",
            "sync_module_states": "True",
            "activation_checkpointing": "False",
            "min_num_params": 1,
        }

    def tearDown(self):
        super().tearDown()

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
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
            self.assertEqual(os.environ.get("ACCELERATE_USE_FSDP", "false"), "true")

    @require_torch_multi_gpu
    @slow
    def test_basic_run(self):
        launcher = get_launcher(distributed=True, use_accelerate=False)
        output_dir = self.get_auto_remove_tmp_dir()
        args = f"""
            --model_name_or_path bert-base-cased
            --task_name mrpc
            --output_dir {output_dir}
            --overwrite_output_dir
            --do_train
            --max_seq_length 128
            --per_device_train_batch_size 16
            --learning_rate 5e-5
            --num_train_epochs 3
            --lr_scheduler_type cosine
            --logging_steps 1
            --save_strategy epoch
            --do_eval
            --evaluation_strategy epoch
            --load_best_model_at_end 
            --skip_memory_metrics False
        """.split()

        fsdp_args = """
            --fsdp "shard_grad_op auto_wrap"  
            --fsdp_transformer_layer_cls_to_wrap BertLayer
        """.split()

        script = [f"{self.examples_dir_str}/pytorch/text-classification/run_glue.py"]
        cmd = launcher + script + args + fsdp_args

        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

    @parameterized.expand(dtypes)
    @require_torch_multi_gpu
    @slow
    def test_basic_run_with_cpu_offload(self, dtype):
        launcher = get_launcher(distributed=True, use_accelerate=False)
        output_dir = self.get_auto_remove_tmp_dir()
        args = f"""
            --model_name_or_path bert-base-cased
            --task_name mrpc
            --output_dir {output_dir}
            --overwrite_output_dir
            --do_train
            --max_seq_length 128
            --per_device_train_batch_size 16
            --learning_rate 5e-5
            --num_train_epochs 3
            --lr_scheduler_type cosine
            --logging_steps 1
            --save_strategy epoch
            --do_eval
            --evaluation_strategy epoch
            --load_best_model_at_end 
            --skip_memory_metrics False
            --{dtype}
        """.split()

        fsdp_args = """
            --fsdp "shard_grad_op auto_wrap offload" 
            --fsdp_transformer_layer_cls_to_wrap BertLayer
        """.split()

        script = [f"{self.examples_dir_str}/pytorch/text-classification/run_glue.py"]
        cmd = launcher + script + args + fsdp_args

        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

    @parameterized.expand(params_with_state_dict_type, name_func=parameterized_custom_name_func)
    @require_torch_multi_gpu
    @slow
    def test_training_and_can_resume_normally(self, sharding_strategy, dtype, state_dict_type):
        output_dir = self.get_auto_remove_tmp_dir("./xxx", after=False)

        if state_dict_type == "LOCAL_STATE_DICT":
            return

        use_accelerate = state_dict_type == "SHARDED_STATE_DICT"
        launcher = get_launcher(True, use_accelerate=use_accelerate)

        args = f"""
            --model_name_or_path bert-base-cased
            --task_name mrpc
            --output_dir {output_dir}
            --overwrite_output_dir
            --do_train
            --max_seq_length 128
            --per_device_train_batch_size 16
            --learning_rate 5e-5
            --num_train_epochs 2
            --lr_scheduler_type cosine
            --logging_steps 1
            --save_strategy epoch
            --do_eval
            --evaluation_strategy epoch
            --{dtype}
        """.split()

        script = [f"{self.examples_dir_str}/pytorch/text-classification/run_glue.py"]

        if not use_accelerate:
            fsdp_args = f"""
                --fsdp "{sharding_strategy} auto_wrap"  
                --fsdp_transformer_layer_cls_to_wrap BertLayer
            """.split()

            cmd = launcher + script + args + fsdp_args
        else:
            fsdp_config = f"""
                --fsdp_sharding_strategy={FSDP_SHARDING_STRATEGY.index(sharding_strategy.upper()) + 1}
            """.split()
            cmd = launcher + fsdp_config + script + args

        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        logs = TrainerState.load_from_json(os.path.join(output_dir, "trainer_state.json")).log_history

        # resume from ckpt
        checkpoint = os.path.join(output_dir, "checkpoint-115")
        resume_args = f"""
            --model_name_or_path bert-base-cased
            --task_name mrpc
            --output_dir {output_dir}
            --overwrite_output_dir
            --do_train
            --max_seq_length 128
            --per_device_train_batch_size 16
            --learning_rate 5e-5
            --num_train_epochs 2
            --lr_scheduler_type cosine
            --logging_steps 1
            --save_strategy epoch
            --do_eval
            --evaluation_strategy epoch
            --{dtype}
            --resume_from_checkpoint {checkpoint}
        """.split()

        if not use_accelerate:
            fsdp_args = f"""
                --fsdp "{sharding_strategy} auto_wrap"  
                --fsdp_transformer_layer_cls_to_wrap BertLayer
            """.split()

            cmd = launcher + script + resume_args + fsdp_args
        else:
            fsdp_config = f"""
                --fsdp_sharding_strategy={FSDP_SHARDING_STRATEGY.index(sharding_strategy.upper()) + 1}
            """.split()
            cmd = launcher + fsdp_config + script + resume_args

        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        logs_resume = TrainerState.load_from_json(os.path.join(output_dir, "trainer_state.json")).log_history

        for log, log1 in zip(logs, logs_resume):
            if "learning_rate" in log:
                self.assertAlmostEqual(log["learning_rate"], log1["learning_rate"], delta=1e-5)
