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

import math
import os
import re
import sys
import unittest
from pathlib import Path
from typing import Tuple
from unittest.mock import patch

from parameterized import parameterized

from transformers.testing_utils import (
    CaptureStderr,
    ExtendSysPath,
    TestCasePlus,
    execute_subprocess_async,
    get_gpu_count,
    get_torch_dist_unique_port,
    require_apex,
    require_bitsandbytes,
    require_fairscale,
    require_torch,
    require_torch_gpu,
    require_torch_multi_gpu,
    require_torch_non_multi_gpu,
    slow,
)
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import set_seed


bindir = os.path.abspath(os.path.dirname(__file__))
with ExtendSysPath(f"{bindir}/../../examples/pytorch/translation"):
    from run_translation import main  # noqa


set_seed(42)
MARIAN_MODEL = "sshleifer/student_marian_en_ro_6_1"
MBART_TINY = "sshleifer/tiny-mbart"


@require_torch
class TestTrainerExt(TestCasePlus):
    def run_seq2seq_quick(
        self,
        distributed=False,
        extra_args_str=None,
        predict_with_generate=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
    ):
        output_dir = self.run_trainer(
            eval_steps=1,
            max_len=12,
            model_name=MBART_TINY,
            num_train_epochs=1,
            distributed=distributed,
            extra_args_str=extra_args_str,
            predict_with_generate=predict_with_generate,
            do_train=do_train,
            do_eval=do_eval,
            do_predict=do_predict,
        )
        logs = TrainerState.load_from_json(os.path.join(output_dir, "trainer_state.json")).log_history

        if not do_eval:
            return

        eval_metrics = [log for log in logs if "eval_loss" in log.keys()]

        first_step_stats = eval_metrics[0]
        if predict_with_generate:
            assert "eval_bleu" in first_step_stats

            last_step_stats = eval_metrics[-1]
            assert isinstance(last_step_stats["eval_bleu"], float)
            assert not math.isnan(float(last_step_stats["eval_loss"])), "eval_loss must not be `nan`"

    @require_torch_non_multi_gpu
    def test_run_seq2seq_no_dist(self):
        self.run_seq2seq_quick()

    # verify that the trainer can handle non-distributed with n_gpu > 1
    @require_torch_multi_gpu
    def test_run_seq2seq_dp(self):
        self.run_seq2seq_quick(distributed=False)

    # verify that the trainer can handle distributed with n_gpu > 1
    @require_torch_multi_gpu
    def test_run_seq2seq_ddp(self):
        self.run_seq2seq_quick(distributed=True)

    # test --sharded_ddp w/o --fp16
    @unittest.skip("Requires an update of the env running those tests")
    @require_torch_multi_gpu
    @require_fairscale
    def test_run_seq2seq_sharded_ddp(self):
        self.run_seq2seq_quick(distributed=True, extra_args_str="--sharded_ddp simple")

    # test --sharded_ddp w/ --fp16
    @unittest.skip("Requires an update of the env running those tests")
    @require_torch_multi_gpu
    @require_fairscale
    def test_run_seq2seq_sharded_ddp_fp16(self):
        self.run_seq2seq_quick(distributed=True, extra_args_str="--sharded_ddp simple --fp16")

    # test --sharded_ddp zero_dp_2 w/o --fp16
    @unittest.skip("Requires an update of the env running those tests")
    @require_torch_multi_gpu
    @require_fairscale
    def test_run_seq2seq_fully_sharded_ddp(self):
        self.run_seq2seq_quick(distributed=True, extra_args_str="--sharded_ddp zero_dp_2", predict_with_generate=False)

    # test --sharded_ddp zero_dp_2 w/ --fp16
    @unittest.skip("Requires an update of the env running those tests")
    @require_torch_multi_gpu
    @require_fairscale
    def test_run_seq2seq_fully_sharded_ddp_fp16(self):
        self.run_seq2seq_quick(
            distributed=True, extra_args_str="--sharded_ddp zero_dp_2 --fp16", predict_with_generate=False
        )

    @require_apex
    @require_torch_gpu
    def test_run_seq2seq_apex(self):
        # XXX: apex breaks the trainer if it's run twice e.g. run_seq2seq.main() from the same
        # program and it breaks other tests that run from the same pytest worker, therefore until this is
        # sorted out it must be run only in an external program, that is distributed=True in this
        # test and only under one or more gpus - if we want cpu will need to make a special test
        #
        # specifically to the problem traced it to self.optimizer.step() - if it's run 2nd time via
        # 2nd main() call it botches the future eval.
        #
        self.run_seq2seq_quick(distributed=True, extra_args_str="--fp16 --fp16_backend=apex")
        # test 2nd time - was getting eval_loss': nan'
        # to reproduce the problem set distributed=False
        self.run_seq2seq_quick(distributed=True, extra_args_str="--fp16 --fp16_backend=apex")

    @parameterized.expand(["base", "low", "high", "mixed"])
    @require_torch_multi_gpu
    def test_trainer_log_level_replica(self, experiment_id):
        # as each sub-test is slow-ish split into multiple sub-tests to avoid CI timeout
        experiments = {
            # test with the default log_level - should be info and thus log info once
            "base": {"extra_args_str": "", "n_matches": 1},
            # test with low log_level and log_level_replica - should be noisy on all processes
            # now the info string should appear twice on 2 processes
            "low": {"extra_args_str": "--log_level debug --log_level_replica debug", "n_matches": 2},
            # test with high log_level and low log_level_replica
            # now the info string should appear once only on the replica
            "high": {"extra_args_str": "--log_level error --log_level_replica debug", "n_matches": 1},
            # test with high log_level and log_level_replica - should be quiet on all processes
            "mixed": {"extra_args_str": "--log_level error --log_level_replica error", "n_matches": 0},
        }

        data = experiments[experiment_id]
        kwargs = {"distributed": True, "predict_with_generate": False, "do_eval": False, "do_predict": False}
        log_info_string = "Running training"
        with CaptureStderr() as cl:
            self.run_seq2seq_quick(**kwargs, extra_args_str=data["extra_args_str"])
        n_matches = len(re.findall(log_info_string, cl.err))
        self.assertEqual(n_matches, data["n_matches"])

    @slow
    def test_run_seq2seq(self):
        output_dir = self.run_trainer(
            eval_steps=2,
            max_len=128,
            model_name=MARIAN_MODEL,
            learning_rate=3e-4,
            num_train_epochs=10,
            distributed=False,
        )

        # Check metrics
        logs = TrainerState.load_from_json(os.path.join(output_dir, "trainer_state.json")).log_history
        eval_metrics = [log for log in logs if "eval_loss" in log.keys()]
        first_step_stats = eval_metrics[0]
        last_step_stats = eval_metrics[-1]

        assert first_step_stats["eval_loss"] > last_step_stats["eval_loss"], "model learned nothing"
        assert isinstance(last_step_stats["eval_bleu"], float)

        # test if do_predict saves generations and metrics
        contents = os.listdir(output_dir)
        contents = {os.path.basename(p) for p in contents}
        assert "generated_predictions.txt" in contents
        assert "predict_results.json" in contents

    @slow
    @require_bitsandbytes
    def test_run_seq2seq_bnb(self):
        from transformers.training_args import OptimizerNames

        def train_and_return_metrics(optim: str) -> Tuple[int, float]:
            extra_args = "--skip_memory_metrics 0"

            output_dir = self.run_trainer(
                max_len=128,
                model_name=MARIAN_MODEL,
                learning_rate=3e-4,
                num_train_epochs=1,
                optim=optim,
                distributed=True,  # force run in a new process
                extra_args_str=extra_args,
                do_eval=False,
                do_predict=False,
                n_gpus_to_use=1,  # to allow deterministic fixed memory usage
            )

            # Check metrics
            logs = TrainerState.load_from_json(Path(output_dir, "trainer_state.json")).log_history
            gpu_peak_mem_mb = int(logs[0]["train_mem_gpu_peaked_delta"] / 2**20)
            gpu_alloc_mem_mb = int(logs[0]["train_mem_gpu_alloc_delta"] / 2**20)

            loss = logs[0]["train_loss"]
            return gpu_peak_mem_mb, gpu_alloc_mem_mb, loss

        gpu_peak_mem_orig, gpu_alloc_mem_orig, loss_orig = train_and_return_metrics(OptimizerNames.ADAMW_TORCH.value)
        gpu_peak_mem_bnb, gpu_alloc_mem_bnb, loss_bnb = train_and_return_metrics(OptimizerNames.ADAMW_BNB.value)

        gpu_alloc_mem_diff = gpu_alloc_mem_orig - gpu_alloc_mem_bnb

        gpu_total_mem_orig = gpu_peak_mem_orig + gpu_alloc_mem_orig
        gpu_total_mem_bnb = gpu_peak_mem_bnb + gpu_alloc_mem_bnb
        gpu_total_mem_diff = gpu_total_mem_orig - gpu_total_mem_bnb

        # sshleifer/student_marian_en_ro_6_1 has 54M parameter, 29M of which is `nn.Embedding` which
        # doesn't get quantized and remains in fp32. Therefore we only have 25M parameters quantized
        # in 2 bytes and the diff in optim memory usage is derived as so:
        #
        # - normal 25*8=~200MB (8 bytes per param)
        # - bnb    25*2= ~50MB (2 bytes per param)
        #
        # Thus we should expect ~150MB total memory saved.
        #
        # Peak memory should be the same - the total should be different by about that same margin
        #
        # After leaving a small margin to accommodate for differences between gpus let's check
        # that we have at least 120MB in savings
        expected_savings = 120

        # uncomment the following if this test starts failing - requires py38 for a new print feature
        # gpu_peak_mem_diff = gpu_peak_mem_orig - gpu_peak_mem_bnb
        # print(f"{gpu_alloc_mem_orig=}MB {gpu_peak_mem_orig=}MB {gpu_alloc_mem_orig+gpu_peak_mem_orig=}MB")
        # print(f" {gpu_alloc_mem_bnb=}MB  {gpu_peak_mem_bnb=}MB  {gpu_alloc_mem_bnb+gpu_peak_mem_bnb=}MB")
        # print(f"{gpu_alloc_mem_diff=}MB")
        # print(f"{gpu_peak_mem_diff=}MB")
        # print(f"{gpu_total_mem_orig=}MB, {gpu_total_mem_bnb=}MB")
        # print(f"{gpu_total_mem_diff=}MB, {gpu_total_mem_diff=}MB")

        self.assertGreater(
            gpu_alloc_mem_diff,
            expected_savings,
            "should use ~150MB less alloc gpu memory with BNB, compared to without it for this model but got"
            f" a difference of {gpu_alloc_mem_diff}MB, with gpu_alloc_mem_orig={gpu_alloc_mem_orig}MB and"
            f" gpu_alloc_mem_bnb={gpu_alloc_mem_bnb}MB",
        )

        self.assertGreater(
            gpu_total_mem_diff,
            expected_savings,
            "should use ~150MB less total gpu memory with BNB, compared to without it for this model but got"
            f" a difference of {gpu_total_mem_diff}MB, with gpu_total_mem_orig={gpu_total_mem_orig}MB and"
            f" gpu_total_mem_bnb={gpu_total_mem_bnb}MB",
        )

        self.assertEqual(
            loss_orig, loss_bnb, f"loss should be the same, but got loss_orig={loss_orig}, loss_bnb={loss_bnb}"
        )

    def run_trainer(
        self,
        max_len: int,
        model_name: str,
        num_train_epochs: int,
        learning_rate: float = 3e-3,
        optim: str = "adafactor",
        distributed: bool = False,
        extra_args_str: str = None,
        eval_steps: int = 0,
        predict_with_generate: bool = True,
        do_train: bool = True,
        do_eval: bool = True,
        do_predict: bool = True,
        n_gpus_to_use: int = None,
    ):
        data_dir = self.test_file_dir / "../fixtures/tests_samples/wmt_en_ro"
        output_dir = self.get_auto_remove_tmp_dir()
        args_train = f"""
            --model_name_or_path {model_name}
            --train_file {data_dir}/train.json
            --validation_file {data_dir}/val.json
            --test_file {data_dir}/test.json
            --output_dir {output_dir}
            --overwrite_output_dir
            --max_train_samples 8
            --max_source_length {max_len}
            --max_target_length {max_len}
            --do_train
            --num_train_epochs {str(num_train_epochs)}
            --per_device_train_batch_size 4
            --learning_rate {learning_rate}
            --warmup_steps 8
            --logging_steps 0
            --logging_strategy no
            --save_steps {str(eval_steps)}
            --group_by_length
            --label_smoothing_factor 0.1
            --target_lang ro_RO
            --source_lang en_XX
        """.split()

        args_eval = f"""
            --do_eval
            --per_device_eval_batch_size 4
            --max_eval_samples 8
            --val_max_target_length {max_len}
            --evaluation_strategy steps
            --eval_steps {str(eval_steps)}
        """.split()

        args_predict = """
            --do_predict
        """.split()

        args = []
        if do_train:
            args += args_train

        if do_eval:
            args += args_eval

        if do_predict:
            args += args_predict

        if predict_with_generate:
            args += "--predict_with_generate".split()

        if do_train:
            if optim == "adafactor":
                args += "--adafactor".split()
            else:
                args += f"--optim {optim}".split()

        if extra_args_str is not None:
            args += extra_args_str.split()

        if distributed:
            if n_gpus_to_use is None:
                n_gpus_to_use = get_gpu_count()
            master_port = get_torch_dist_unique_port()
            distributed_args = f"""
                -m torch.distributed.launch
                --nproc_per_node={n_gpus_to_use}
                --master_port={master_port}
                {self.examples_dir_str}/pytorch/translation/run_translation.py
            """.split()
            cmd = [sys.executable] + distributed_args + args
            # keep for quick debug
            # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
            execute_subprocess_async(cmd, env=self.get_env())
        else:
            testargs = ["run_translation.py"] + args
            with patch.object(sys, "argv", testargs):
                main()

        return output_dir
