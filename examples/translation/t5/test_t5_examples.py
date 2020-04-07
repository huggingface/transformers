import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .evaluate_wmt import run_generate


text = ["When Liana Barrientos was 23 years old, she got married in Westchester County."]
translation = ["Als Liana Barrientos 23 Jahre alt war, heiratete sie in Westchester County."]

output_file_name = "output_t5_trans.txt"
score_file_name = "score_t5_trans.txt"

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


class TestT5Examples(unittest.TestCase):
    def test_t5_cli(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_source = Path(tempfile.gettempdir()) / "utest_generations_t5_trans.hypo"
        with tmp_source.open("w") as f:
            f.write("\n".join(text))

        tmp_target = Path(tempfile.gettempdir()) / "utest_generations_t5_trans.target"
        with tmp_target.open("w") as f:
            f.write("\n".join(translation))

        output_file_name = Path(tempfile.gettempdir()) / "utest_output_trans.hypo"
        score_file_name = Path(tempfile.gettempdir()) / "utest_score.hypo"

        testargs = [
            "evaluate_wmt.py",
            "patrickvonplaten/t5-tiny-random",
            str(tmp_source),
            str(output_file_name),
            str(tmp_target),
            str(score_file_name),
        ]

        with patch.object(sys, "argv", testargs):
            run_generate()
            self.assertTrue(Path(output_file_name).exists())
            self.assertTrue(Path(score_file_name).exists())
