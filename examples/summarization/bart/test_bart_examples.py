import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from torch.utils.data import DataLoader

from transformers import BartTokenizer

from .evaluate_cnn import _run_generate
from .utils import SummarizationDataset


output_file_name = "output_bart_sum.txt"

articles = [" New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County."]

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
CNN_TINY_PATH = "/Users/shleifer/Dropbox/cnn_tinier"


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

    def test_bart_summarization_dataset(self):
        tokenizer = BartTokenizer.from_pretrained("bart-large")
        train_dataset = SummarizationDataset(
            tokenizer, data_dir=CNN_TINY_PATH, type_path="train", max_source_length=1024, max_target_length=1024
        )
        dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=train_dataset.collate_fn)
        for batch in dataloader:
            print({k: x.shape for k, x in batch.items()})
