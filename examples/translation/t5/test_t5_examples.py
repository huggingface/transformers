import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .evaluate_wmt import run_generate


text = [" When Liana Barrientos was 23 years old, she got married in Westchester County."]
translation = [" Als Liana Barrientos 23 Jahre alt war, heiratete sie in Westchester County."]

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


class TestT5Examples(unittest.TestCase):
    def test_t5_cli(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_source = Path(tempfile.gettempdir()) / "utest_generations.hypo"
        with tmp_source.open("w") as f:
            f.write("\n".join(text))

        tmp_target = Path(tempfile.gettempdir()) / "utest_generations.hypo"
        with tmp_target.open("w") as f:
            f.write("\n".join(text))

        testargs = ["evaluate_cnn.py", str(tmp_source), "output.txt", str(tmp_target), "score.txt"]

        with patch.object(sys, "argv", testargs):
            run_generate()
            self.assertTrue(Path("output.txt").exists())
