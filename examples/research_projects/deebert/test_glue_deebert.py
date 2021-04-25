import argparse
import logging
import sys
from unittest.mock import patch

import run_glue_deebert
from transformers.testing_utils import TestCasePlus, get_gpu_count, require_torch_non_multi_gpu, slow


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def get_setup_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f")
    args = parser.parse_args()
    return args.f


class DeeBertTests(TestCasePlus):
    def setup(self) -> None:
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

    def run_and_check(self, args):
        n_gpu = get_gpu_count()

        if n_gpu > 1:
            pass
            # XXX: doesn't quite work with n_gpu > 1 https://github.com/huggingface/transformers/issues/10560
            # script = f"{self.examples_dir_str}/research_projects/deebert/run_glue_deebert.py"
            # distributed_args = f"-m torch.distributed.launch --nproc_per_node={n_gpu} {script}".split()
            # cmd = [sys.executable] + distributed_args + args
            # execute_subprocess_async(cmd, env=self.get_env())
            # XXX: test the results - need to save them first into .json file
        else:
            args.insert(0, "run_glue_deebert.py")
            with patch.object(sys, "argv", args):
                result = run_glue_deebert.main()
                for value in result.values():
                    self.assertGreaterEqual(value, 0.666)

    @slow
    @require_torch_non_multi_gpu
    def test_glue_deebert_train(self):

        train_args = """
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
        self.run_and_check(train_args)

        eval_args = """
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
        self.run_and_check(eval_args)

        entropy_eval_args = """
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
        self.run_and_check(entropy_eval_args)
