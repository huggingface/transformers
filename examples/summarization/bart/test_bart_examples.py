import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .evaluate_cnn import DEFAULT_DEVICE, _run_generate


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

    @unittest.skipUnless(DEFAULT_DEVICE == "cuda", "requires GPU")
    def test_bart_run_sum_cli(self):
        cmd = "./run_train_tiny.sh"
        os.system(cmd)
