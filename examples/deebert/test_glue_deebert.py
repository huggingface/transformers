import argparse
import logging
import sys
import unittest
from unittest.mock import patch

import run_glue_deebert
from transformers.testing_utils import slow


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def get_setup_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f")
    args = parser.parse_args()
    return args.f


class DeeBertTests(unittest.TestCase):
    def setup(self) -> None:
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

    @slow
    def test_glue_deebert_train(self):

        train_args = """
            run_glue_deebert.py
            --model_type roberta
            --model_name_or_path roberta-base
            --task_name MRPC
            --do_train
            --do_eval
            --do_lower_case
            --data_dir ./tests/fixtures/tests_samples/MRPC/
            --max_seq_length 128
            --per_gpu_eval_batch_size=1
            --per_gpu_train_batch_size=8
            --learning_rate 2e-4
            --num_train_epochs 3
            --overwrite_output_dir
            --seed 42
            --output_dir ./examples/deebert/saved_models/roberta-base/MRPC/two_stage
            --plot_data_dir ./examples/deebert/results/
            --save_steps 0
            --overwrite_cache
            --eval_after_first_stage
            """.split()
        with patch.object(sys, "argv", train_args):
            result = run_glue_deebert.main()
            for value in result.values():
                self.assertGreaterEqual(value, 0.666)

        eval_args = """
            run_glue_deebert.py
            --model_type roberta
            --model_name_or_path ./examples/deebert/saved_models/roberta-base/MRPC/two_stage
            --task_name MRPC
            --do_eval
            --do_lower_case
            --data_dir ./tests/fixtures/tests_samples/MRPC/
            --output_dir ./examples/deebert/saved_models/roberta-base/MRPC/two_stage
            --plot_data_dir ./examples/deebert/results/
            --max_seq_length 128
            --eval_each_highway
            --eval_highway
            --overwrite_cache
            --per_gpu_eval_batch_size=1
            """.split()
        with patch.object(sys, "argv", eval_args):
            result = run_glue_deebert.main()
            for value in result.values():
                self.assertGreaterEqual(value, 0.666)

        entropy_eval_args = """
            run_glue_deebert.py
            --model_type roberta
            --model_name_or_path ./examples/deebert/saved_models/roberta-base/MRPC/two_stage
            --task_name MRPC
            --do_eval
            --do_lower_case
            --data_dir ./tests/fixtures/tests_samples/MRPC/
            --output_dir ./examples/deebert/saved_models/roberta-base/MRPC/two_stage
            --plot_data_dir ./examples/deebert/results/
            --max_seq_length 128
            --early_exit_entropy 0.1
            --eval_highway
            --overwrite_cache
            --per_gpu_eval_batch_size=1
            """.split()
        with patch.object(sys, "argv", entropy_eval_args):
            result = run_glue_deebert.main()
            for value in result.values():
                self.assertGreaterEqual(value, 0.666)
