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
    mockenv,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
)
from transformers.trainer_utils import set_seed


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


@slow
@require_deepspeed
@require_torch_gpu
class TestDeepSpeed(TestCasePlus):

    # this setup emulates a notebook where a launcher needs to be emulated by hand
    @mockenv(MASTER_ADDR="localhost", MASTER_PORT="10999", RANK="0", LOCAL_RANK="0", WORLD_SIZE="1")
    def test_fake_notebook_no_launcher(self):
        sys.path.append(self.tests_dir_str)
        from test_trainer import get_regression_trainer

        del sys.path[-1]  # restore
        ds_config_file = f"{self.test_file_dir_str}/ds_config.json"
        with CaptureStd() as cs:
            trainer = get_regression_trainer(local_rank=0, deepspeed=ds_config_file)
            trainer.train()
        assert "DeepSpeed info" in cs.out, "expected DeepSpeed logger output but got none"

    @require_torch_multi_gpu
    def test_basic_distributed(self):
        self.run_quick(distributed=True)

    @require_torch_multi_gpu
    def test_grad_acum(self):
        self.run_quick(distributed=True, extra_args_str="--gradient_accumulation_steps 2")

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
            --task translation
            --target_lang ro_RO
            --source_lang en_XX
        """.split()

        if extra_args_str is not None:
            args.extend(extra_args_str.split())

        if remove_args_str is not None:
            remove_args = remove_args_str.split()
            args = [x for x in args if x not in remove_args]

        ds_args = f"--deepspeed {self.test_file_dir_str}/ds_config.json".split()
        script = [f"{self.examples_dir_str}/seq2seq/run_seq2seq.py"]
        num_gpus = get_gpu_count() if distributed else 1
        launcher = f"deepspeed --num_gpus {num_gpus}".split()

        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"PYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        return output_dir
