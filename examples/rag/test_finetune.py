import logging
import os
import sys
from pathlib import Path
from unittest.mock import patch

import torch

from transformers.file_utils import is_apex_available
from transformers.testing_utils import TestCasePlus, require_torch_gpu, require_torch_multigpu


sys.path.append(str(Path(__file__).parent.resolve()))
import finetune  # noqa


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


class RagFinetuneExampleTests(TestCasePlus):
    def _run_finetune(self, gpus: int):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        data_dir = Path(__file__).parent / "test_data" / "dummy_seq2seq"
        data_dir = str(data_dir.resolve())
        # Note:
        # We use the ddp_spawn backend here because of a memory issue with ddp.
        # Indeed ddp doesn't seem to always unload models between fit and test (as of pl 1.0.6)
        # which causes memory to fill up.
        testargs = f"""
            finetune.py \
                --data_dir {data_dir} \
                --output_dir {tmp_dir} \
                --model_name_or_path facebook/rag-sequence-base \
                --model_type rag_sequence \
                --do_train \
                --do_predict \
                --n_val -1 \
                --val_check_interval 1.0 \
                --train_batch_size 2 \
                --eval_batch_size 1 \
                --max_source_length 25 \
                --max_target_length 25 \
                --val_max_target_length 25 \
                --test_max_target_length 25 \
                --label_smoothing 0.1 \
                --dropout 0.1 \
                --attention_dropout 0.1 \
                --weight_decay 0.001 \
                --adam_epsilon 1e-08 \
                --max_grad_norm 0.1 \
                --lr_scheduler polynomial \
                --learning_rate 3e-04 \
                --num_train_epochs 1 \
                --warmup_steps 4 \
                --gradient_accumulation_steps 1 \
                --distributed-port 8888 \
                --use_dummy_dataset 1 \
                --accelerator ddp_spawn
            """.split()

        if gpus > 0:
            testargs.append(f"--gpus={gpus}")
            if is_apex_available():
                testargs.append("--fp16")
        else:
            testargs.append("--gpus=0")
            testargs.append("--distributed_backend=ddp_cpu")
            testargs.append("--num_processes=2")

        with patch.object(sys, "argv", testargs):
            result = finetune.main()[0]
            return result

    @require_torch_gpu
    def test_finetune_gpu(self):
        import json

        result = self._run_finetune(gpus=1)
        print(result)
        open("out.txt", "w").write(json.dumps(result))
        self.assertGreaterEqual(result["test_em"], 0.2)

    @require_torch_multigpu
    def test_finetune_multigpu(self):
        import json

        import filelock

        filelock.logger().setLevel(level=logging.ERROR)

        result = self._run_finetune(gpus=2)
        open("out.txt", "w").write(json.dumps(result))
        self.assertGreaterEqual(result["test_em"], 0.2)
