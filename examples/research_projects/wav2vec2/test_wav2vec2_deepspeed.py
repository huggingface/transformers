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


# XXX: we want transformers master here - in the absense of conftest manipulating sys.path:
# hack it in for now:
import sys
from pathlib import Path


git_repo_path = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(1, str(git_repo_path))

import dataclasses  # noqa
import io  # noqa
import itertools  # noqa
import json  # noqa
import os  # noqa
import unittest  # noqa
from copy import deepcopy  # noqa

from parameterized import parameterized  # noqa
from transformers import TrainingArguments, is_torch_available  # noqa
from transformers.deepspeed import is_deepspeed_available  # noqa
from transformers.file_utils import WEIGHTS_NAME  # noqa
from transformers.testing_utils import (  # noqa
    CaptureLogger,
    ExtendSysPath,
    TestCasePlus,
    execute_subprocess_async,
    get_gpu_count,
    mockenv_context,
    require_deepspeed,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
)
from transformers.trainer_utils import set_seed  # noqa


set_seed(42)

models = dict(base="patrickvonplaten/wav2vec2_tiny_random", robust="patrickvonplaten/wav2vec2_tiny_random_robust")

ZERO2 = "zero2"
ZERO3 = "zero3"
stages = [ZERO2, ZERO3]


def custom_name_func(func, param_num, param):
    # customize the test name generator function as we want both params to appear in the sub-test
    # name, as by default it shows only the first param
    param_based_name = parameterized.to_safe_name("_".join(str(x) for x in param.args))
    return f"{func.__name__}_{param_based_name}"


# Cartesian-product of zero stages with models to test
params = list(itertools.product(stages, models.keys()))


@slow
@require_deepspeed
@require_torch_gpu
class TestDeepSpeedWav2Vec2(TestCasePlus):
    @parameterized.expand(params, name_func=custom_name_func)
    def test_fp32_non_distributed(self, stage, model):
        self.run_and_check(
            stage=stage,
            model=model,
            distributed=False,
            fp16=False,
        )

    @require_torch_multi_gpu
    @parameterized.expand(params, name_func=custom_name_func)
    def test_fp32_distributed(self, stage, model):
        self.run_and_check(
            stage=stage,
            model=model,
            distributed=True,
            fp16=False,
        )

    @parameterized.expand(params, name_func=custom_name_func)
    def test_fp16_non_distributed(self, stage, model):
        self.run_and_check(
            stage=stage,
            model=model,
            distributed=False,
            fp16=True,
        )

    @require_torch_multi_gpu
    @parameterized.expand(params, name_func=custom_name_func)
    def test_fp16_distributed(self, stage, model):
        self.run_and_check(
            stage=stage,
            model=model,
            distributed=True,
            fp16=True,
        )

    def do_checks(self, output_dir):
        # XXX: run_asr is premature and doesn't save any results
        # so all we check for now is that the process didn't fail
        pass

    # XXX: need to do better validation beyond just that the run was successful
    def run_and_check(
        self,
        stage: str,
        model: str,
        eval_steps: int = 10,
        distributed: bool = True,
        quality_checks: bool = True,
        fp16: bool = True,
    ):

        model_name = models[model]

        output_dir = self.run_trainer(
            stage=stage,
            model_name=model_name,
            eval_steps=eval_steps,
            num_train_epochs=1,
            distributed=distributed,
            fp16=fp16,
        )

        self.do_checks(output_dir)

        return output_dir

    def run_trainer(
        self,
        stage: str,
        model_name: str,
        eval_steps: int = 10,
        num_train_epochs: int = 1,
        distributed: bool = True,
        fp16: bool = True,
    ):

        output_dir = self.get_auto_remove_tmp_dir("./xxx", after=False)
        args = f"""
            --model_name_or_path {model_name}
            --dataset_name patrickvonplaten/librispeech_asr_dummy
            --dataset_config_name clean
            --train_split_name validation
            --validation_split_name validation
            --output_dir {output_dir}
            --num_train_epochs {str(num_train_epochs)}
            --per_device_train_batch_size 2
            --per_device_eval_batch_size 2
            --evaluation_strategy steps
            --learning_rate 5e-4
            --warmup_steps 8
            --orthography timit
            --preprocessing_num_workers 1
            --group_by_length
            --freeze_feature_extractor
            --report_to none
            --save_steps 0
            --eval_steps {eval_steps}
            --report_to none
        """.split()

        if fp16:
            args.extend(["--fp16"])

        # currently ds_config_wav2vec2_zero.json requires "zero_optimization.find_unused_parameters": true,
        # hence the separate config files
        ds_args = f"--deepspeed {self.test_file_dir_str}/ds_config_wav2vec2_{stage}.json".split()
        script = [f"{self.examples_dir_str}/research_projects/wav2vec2/run_asr.py"]
        launcher = self.get_launcher(distributed)

        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        return output_dir

    def get_launcher(self, distributed=False):
        # 1. explicitly set --num_nodes=1 just in case these tests end up run on a multi-node setup
        # - it won't be able to handle that
        # 2. for now testing with just 2 gpus max (since some quality tests may give different
        # results with mode gpus because we use very little data)
        num_gpus = min(2, get_gpu_count()) if distributed else 1
        return f"deepspeed --num_nodes 1 --num_gpus {num_gpus}".split()
