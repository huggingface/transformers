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
from transformers.file_utils import WEIGHTS_NAME
from transformers.oslo import HfOsloConfig, is_oslo_available
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


def get_master_port(real_launcher=False) -> str:
    # from test_deepspeed.py
    master_port_base = os.environ.get("OSLO_TEST_PORT", DEFAULT_MASTER_PORT)
    if not real_launcher:
        master_port_base = str(int(master_port_base) + 1)
    return master_port_base


def require_oslo(test_case):
    # TODO: move to testing_utils.py once done
    """
    Decorator marking a test that requires oslo
    """
    if not is_oslo_available():
        return unittest.skip("test requires oslo")(test_case)
    return test_case


def get_torch_distributed_launcher():
    num_gpus = min(2, get_gpu_count())
    master_port = get_master_port(real_launcher=True)
    return f"python -m torch.distributed.launch --nproc_per_node {num_gpus} --master_port {master_port}".split()


@slow
@require_oslo
@require_torch_gpu
@require_torch_multi_gpu
class TestOsloWithLauncher(TestCasePlus):
    """This class is for testing via an external script - can do multiple gpus"""

    def test_do_eval_no_train(self):
        # testing only zero3 since zero2 makes no sense with inference
        self.run_and_check(
            eval_steps=1,
            do_train=False,
            do_eval=True,
        )

    def test_fp32(self):
        self.run_and_check(
            model_name=T5_TINY,
            do_train=True,
            do_eval=True,
            quality_checks=False,
            fp16=False,
        )

    def test_deepspeed(self):
        self.run_and_check(
            model_name=T5_TINY,
            do_train=True,
            do_eval=True,
            quality_checks=False,
            fp16=False,
        )

    @parameterized.expand(["fp16", "fp32"])
    def test_inference(self, dtype):
        # this is just inference, so no optimizer should be loaded
        # it only works for z3 (makes no sense with z1-z2)
        fp16 = True if dtype == "fp16" else False
        self.run_and_check(
            model_name=T5_TINY,
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
        model_name: str = T5_SMALL,
        eval_steps: int = 10,
        do_train: bool = True,
        do_eval: bool = True,
        quality_checks: bool = True,
        fp16: bool = True,
        extra_args_str: str = None,
        remove_args_str: str = None,
    ):

        # we are doing quality testing so using a small real model
        output_dir = self.run_trainer(
            model_name=model_name,
            eval_steps=eval_steps,
            num_train_epochs=1,
            do_train=do_train,
            do_eval=do_eval,
            fp16=fp16,
            extra_args_str=extra_args_str,
            remove_args_str=remove_args_str,
        )

        self.do_checks(output_dir, do_train=do_train, do_eval=do_eval, quality_checks=quality_checks)

        return output_dir

    def run_trainer(
        self,
        model_name: str,
        eval_steps: int = 10,
        num_train_epochs: int = 1,
        do_train: bool = False,
        do_eval: bool = True,
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

        oslo_args = f"--oslo {self.test_file_dir_str}/oslo_config.json".split()
        script = [f"{self.examples_dir_str}/pytorch/translation/run_translation.py"]
        launcher = get_torch_distributed_launcher()

        cmd = launcher + script + args + oslo_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        return output_dir

    def test_clm(self):
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

        oslo_args = f"--oslo {self.test_file_dir_str}/oslo_config.json".split()
        script = [f"{self.examples_dir_str}/pytorch/language-modeling/run_clm.py"]
        launcher = get_torch_distributed_launcher()

        cmd = launcher + script + args + oslo_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())
