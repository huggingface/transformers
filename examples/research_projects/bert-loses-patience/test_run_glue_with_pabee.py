import argparse
import logging
import sys
from unittest.mock import patch

import run_glue_with_pabee

from transformers.testing_utils import TestCasePlus


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def get_setup_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f")
    args = parser.parse_args()
    return args.f


class PabeeTests(TestCasePlus):
    def test_run_glue(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_glue_with_pabee.py
            --model_type albert
            --model_name_or_path albert/albert-base-v2
            --data_dir ./tests/fixtures/tests_samples/MRPC/
            --output_dir {tmp_dir}
            --overwrite_output_dir
            --task_name mrpc
            --do_train
            --do_eval
            --per_gpu_train_batch_size=2
            --per_gpu_eval_batch_size=1
            --learning_rate=2e-5
            --max_steps=50
            --warmup_steps=2
            --seed=42
            --max_seq_length=128
            """.split()

        with patch.object(sys, "argv", testargs):
            result = run_glue_with_pabee.main()
            for value in result.values():
                self.assertGreaterEqual(value, 0.75)
