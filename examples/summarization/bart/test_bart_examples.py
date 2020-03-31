import argparse
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .evaluate_cnn import _run_generate
from .run_bart_sum import main


output_file_name = "output_bart_sum.txt"

articles = [" New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County."]

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()

DEFAULT_ARGS = {
    "output_dir": "",
    "fp16": False,
    "fp16_opt_level": "O1",
    "n_gpu": 1,
    "n_tpu_cores": 0,
    "max_grad_norm": 1.0,
    "do_train": True,
    "do_predict": False,
    "gradient_accumulation_steps": 1,
    "server_ip": "",
    "server_port": "",
    "seed": 42,
    "model_type": "bart",
    "model_name_or_path": "sshleifer/bart-tiny-random",
    "config_name": "",
    "tokenizer_name": "",
    "cache_dir": "",
    "do_lower_case": False,
    "learning_rate": 3e-05,
    "weight_decay": 0.0,
    "adam_epsilon": 1e-08,
    "warmup_steps": 0,
    "num_train_epochs": 1,
    "train_batch_size": 2,
    "eval_batch_size": 2,
    "max_seq_length": 1024,
}


class TestBartExamples(unittest.TestCase):
    def test_bart_cnn_cli(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        tmp = Path(tempfile.gettempdir()) / "utest_generations_bart_sum.hypo"
        with tmp.open("w") as f:
            f.write("\n".join(articles))

        testargs = ["evaluate_cnn.py", str(tmp), output_file_name, "sshleifer/bart-tiny-random"]
        with patch.object(sys, "argv", testargs):
            _run_generate()
            self.assertTrue(Path(output_file_name).exists())
            os.remove(Path(output_file_name))

    def test_bart_run_sum_cli(self):
        # script = 'examples/summarization/bart/run_train_tiny.sh'

        args = """
        --data_dir=cnn_tiny/ \
        --model_type=bart \
        --model_name_or_path=sshleifer/bart-tiny-random \
        --learning_rate=3e-5 \
        --train_batch_size=2 \
        --eval_batch_size=2 \
        --output_dir=$OUTPUT_DIR \
        --num_train_epochs=1  \
        --n_gpu={n_gpu} \
        --do_train
        """.split()
        n_gpu = 0
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        CNN_TINY_PATH = "/Users/shleifer/Dropbox/cnn_tiny/"
        args_d: dict = DEFAULT_ARGS.copy()

        args_d.update(
            data_dir=CNN_TINY_PATH,
            model_type="bart",
            train_batch_size=2,
            eval_batch_size=2,
            n_gpu=n_gpu,
            output_dir="dringus",
        )
        args = argparse.Namespace(**args_d)
        os.makedirs(args.output_dir, exist_ok=False)
        main(args)

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil

        shutil.rmtree("dringus")
