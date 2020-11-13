import logging
import os
import sys
from pathlib import Path
from unittest.mock import patch

import torch

from transformers.file_utils import is_apex_available
from transformers.testing_utils import TestCasePlus

sys.path.append(os.path.dirname(__file__))
import finetune  # noqa

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


class RagFinetuneExampleTests(TestCasePlus):

    def test_finetune(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        data_dir = Path(__file__).parent / "test_data" / "dummy_seq2seq"
        data_dir = str(data_dir.resolve())
        testargs = f"""
            finetune.py \
                --data_dir {data_dir} \
                --output_dir {tmp_dir} \
                --model_name_or_path facebook/rag-sequence-base \
                --model_type rag_sequence \
                --do_train \
                --do_predict \
                --n_val -1 \
                --val_check_interval 0.25 \
                --train_batch_size 4 \
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
                --learning_rate 3e-05 \
                --num_train_epochs 5 \
                --warmup_steps 5 \
                --gradient_accumulation_steps 1 \
                --use_dummy_dataset 1
            """.split()

        # TODO: test with at least two processes instead of 1
        if torch.cuda.is_available():
            testargs.append("--gpus=1")
            if is_apex_available():
                testargs.append("--fp16")
        else:
            testargs.append("--gpus=0")
            testargs.append("--distributed_backend=ddp_cpu")
            testargs.append("--num_processes=1")

        with patch.object(sys, "argv", testargs):
            result = finetune.main()[0]
            import json
            print(result)
            open("out.txt", "w").write(json.dumps(result))
            self.assertGreaterEqual(result["test_em"], 0.2)
