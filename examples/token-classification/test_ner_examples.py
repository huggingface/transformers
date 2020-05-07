import logging
import sys
import unittest
from unittest.mock import patch

import run_ner


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


class ExamplesTests(unittest.TestCase):
    def test_run_ner(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        testargs = """
            --model_name distilbert-base-german-cased
            --output_dir ./tests/fixtures/tests_samples/temp_dir
            --overwrite_output_dir
            --data_dir ./tests/fixtures/tests_samples/GermEval
            --labels ./tests/fixtures/tests_samples/GermEval/labels.txt
            --max_seq_length 128
            --num_train_epochs 6
            --logging_steps 1
            --do_train
            --do_eval
            """.split()
        with patch.object(sys, "argv", ["run.py"] + testargs):
            result = run_ner.main()
            self.assertLess(result["loss"], 1.5)
