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
        return f"accelerate launch --num_processes={num_gpus} --main_process_port={master_port}".split()
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
        output_dir = self.get_auto_remove_tmp_dir("./xxx", after=False)
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
            trainer.train()

    def test_basic_run(self):
        pass
