import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .evaluate_cnn import DEFAULT_DEVICE, _run_generate
from .run_bart_sum import main


output_file_name = "output_bart_sum.txt"

articles = [" New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County."]

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


class TestBartExamples(unittest.TestCase):
    def test_bart_cnn_cli(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        tmp = Path(tempfile.gettempdir()) / "utest_generations_bart_sum.hypo"
        with tmp.open("w") as f:
            f.write("\n".join(articles))
        testargs = ["evaluate_cnn.py", str(tmp), output_file_name]
        with patch.object(sys, "argv", testargs):
            _run_generate()
            self.assertTrue(Path(output_file_name).exists())
            os.remove(Path(output_file_name))


    def test_bart_run_sum_cli(self):
        #script = 'examples/summarization/bart/run_train_tiny.sh'

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
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        tmp = Path(tempfile.gettempdir()) / "utest_generations_bart_sum.hypo"
        #with tmp.open("w") as f:
            #f.write("\n".join(articles))
        
        testargs = ["run_bart_sum.py", + args
        with patch.object(sys, "argv", testargs):
            _run_generate()
            self.assertTrue(Path(output_file_name).exists())
            os.remove(Path(output_file_name))



        self.assertTrue(os.path.exists(script))
        n_gpu = 0 if DEFAULT_DEVICE == 'cpu' else 1
        cmd = f"{script} {n_gpu}"
        os.system(cmd)
