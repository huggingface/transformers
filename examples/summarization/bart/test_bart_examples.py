import argparse
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import evaluate_cnn
from .evaluate_cnn import generate_summaries


articles = [" New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, "]

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


class TestBartExamples(unittest.TestCase):
    def test_bart_cnn_cli(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        tmp = Path(tempfile.gettempdir()) / "utest_generations.hypo"
        with tmp.open("w") as f:
            f.write("\n".join(articles))
        testargs = [
            "evaluate_cnn.py",
            str(tmp),
            "output.txt",
        ]
        # model_type, model_name = ("--model_type=openai-gpt", "--model_name_or_path=openai-gpt")
        with patch.object(sys, "argv", testargs):
            result = evaluate_cnn.main()
            # self.assertGreaterEqual(len(result[0]), 10)


if __name__ == "__main__":
    unittest.main()
