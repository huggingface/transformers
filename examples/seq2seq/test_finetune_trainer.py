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

import os
import sys
import unittest
from unittest.mock import patch

from transformers.file_utils import is_apex_available
from transformers.integrations import is_fairscale_available
from transformers.testing_utils import (
    TestCasePlus,
    execute_subprocess_async,
    get_gpu_count,
    require_torch_multi_gpu,
    require_torch_non_multi_gpu,
    slow,
)
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import set_seed

from .finetune_trainer import main


set_seed(42)
MARIAN_MODEL = "sshleifer/student_marian_en_ro_6_1"
MBART_TINY = "sshleifer/tiny-mbart"


# a candidate for testing_utils
def require_fairscale(test_case):
    """
    Decorator marking a test that requires fairscale
    """
    if not is_fairscale_available():
        return unittest.skip("test requires fairscale")(test_case)
    else:
        return test_case


# a candidate for testing_utils
def require_apex(test_case):
    """
    Decorator marking a test that requires apex
    """
    if not is_apex_available():
        return unittest.skip("test requires apex")(test_case)
    else:
        return test_case


class TestFinetuneTrainer(TestCasePlus):
    def finetune_trainer_quick(self, distributed=None, extra_args_str=None):
        output_dir = self.run_trainer(1, "12", MBART_TINY, 1, distributed, extra_args_str)
        logs = TrainerState.load_from_json(os.path.join(output_dir, "trainer_state.json")).log_history
        eval_metrics = [log for log in logs if "eval_loss" in log.keys()]
        first_step_stats = eval_metrics[0]
        assert "eval_bleu" in first_step_stats

    @require_torch_non_multi_gpu
    def test_finetune_trainer_no_dist(self):
        self.finetune_trainer_quick()

    # the following 2 tests verify that the trainer can handle distributed and non-distributed with n_gpu > 1
    @require_torch_multi_gpu
    def test_finetune_trainer_dp(self):
        self.finetune_trainer_quick(distributed=False)

    @require_torch_multi_gpu
    def test_finetune_trainer_ddp(self):
        self.finetune_trainer_quick(distributed=True)

    # it's crucial to test --sharded_ddp w/ and w/o --fp16
    @require_torch_multi_gpu
    @require_fairscale
    def test_finetune_trainer_ddp_sharded_ddp(self):
        self.finetune_trainer_quick(distributed=True, extra_args_str="--sharded_ddp")

    @require_torch_multi_gpu
    @require_fairscale
    def test_finetune_trainer_ddp_sharded_ddp_fp16(self):
        self.finetune_trainer_quick(distributed=True, extra_args_str="--sharded_ddp --fp16")

    @require_apex
    def test_finetune_trainer_apex(self):
        self.finetune_trainer_quick(extra_args_str="--fp16 --fp16_backend=apex")

    @slow
    def test_finetune_trainer_slow(self):
        # There is a missing call to __init__process_group somewhere
        output_dir = self.run_trainer(
            eval_steps=2, max_len="128", model_name=MARIAN_MODEL, num_train_epochs=10, distributed=False
        )

        # Check metrics
        logs = TrainerState.load_from_json(os.path.join(output_dir, "trainer_state.json")).log_history
        eval_metrics = [log for log in logs if "eval_loss" in log.keys()]
        first_step_stats = eval_metrics[0]
        last_step_stats = eval_metrics[-1]

        assert first_step_stats["eval_bleu"] < last_step_stats["eval_bleu"]  # model learned nothing
        assert isinstance(last_step_stats["eval_bleu"], float)

        # test if do_predict saves generations and metrics
        contents = os.listdir(output_dir)
        contents = {os.path.basename(p) for p in contents}
        assert "test_generations.txt" in contents
        assert "test_results.json" in contents

    def run_trainer(
        self,
        eval_steps: int,
        max_len: str,
        model_name: str,
        num_train_epochs: int,
        distributed: bool = False,
        extra_args_str: str = None,
    ):
        data_dir = self.examples_dir / "seq2seq/test_data/wmt_en_ro"
        output_dir = self.get_auto_remove_tmp_dir()
        args = f"""
            --model_name_or_path {model_name}
            --data_dir {data_dir}
            --output_dir {output_dir}
            --overwrite_output_dir
            --n_train 8
            --n_val 8
            --max_source_length {max_len}
            --max_target_length {max_len}
            --val_max_target_length {max_len}
            --do_train
            --do_eval
            --do_predict
            --num_train_epochs {str(num_train_epochs)}
            --per_device_train_batch_size 4
            --per_device_eval_batch_size 4
            --learning_rate 3e-3
            --warmup_steps 8
            --evaluation_strategy steps
            --predict_with_generate
            --logging_steps 0
            --save_steps {str(eval_steps)}
            --eval_steps {str(eval_steps)}
            --sortish_sampler
            --label_smoothing_factor 0.1
            --adafactor
            --task translation
            --tgt_lang ro_RO
            --src_lang en_XX
        """.split()
        # --eval_beams  2

        if extra_args_str is not None:
            args.extend(extra_args_str.split())

        if distributed:
            n_gpu = get_gpu_count()
            distributed_args = f"""
                -m torch.distributed.launch
                --nproc_per_node={n_gpu}
                {self.test_file_dir}/finetune_trainer.py
            """.split()
            cmd = [sys.executable] + distributed_args + args
            execute_subprocess_async(cmd, env=self.get_env())
        else:
            testargs = ["finetune_trainer.py"] + args
            with patch.object(sys, "argv", testargs):
                main()

        return output_dir
