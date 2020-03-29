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


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


class TestBartExamples(unittest.TestCase):
    def test_bart_cnn_cli(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        tmp = Path(tempfile.gettempdir()) / "utest_generations_bart_sum.hypo"
        output_file_name = "output_bart_sum.txt"
        articles = [" New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County."]
        with tmp.open("w") as f:
            f.write("\n".join(articles))
        testargs = ["evaluate_cnn.py", str(tmp), output_file_name]
        with patch.object(sys, "argv", testargs):
            _run_generate()
            self.assertTrue(Path(output_file_name).exists())
            os.remove(Path(output_file_name))

    def test_bart_summarization_dataset(self):
        tmp_dir = Path(tempfile.gettempdir())
        articles = [" Sam ate lunch today", "Sams lunch ingredients"]
        summaries = ["A very interesting story about what I ate for lunch.", "Avocado, celery, turkey, coffee"]

        with (tmp_dir / "train.source").open("w") as f:
            f.write("\n".join(articles))
        with (tmp_dir / "train.target").open("w") as f:
            f.write("\n".join(summaries))

        tokenizer = BartTokenizer.from_pretrained("bart-large")
        train_dataset = SummarizationDataset(
            tokenizer, data_dir=tmp_dir, type_path="train", max_source_length=1024, max_target_length=1024
        )
        dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=train_dataset.collate_fn)
        for batch in dataloader:
            self.assertEqual(batch["source_mask"].shape, batch["source_ids"].shape)
            self.assertEqual(batch["target_ids"].shape[0], 2)
            self.assertGreaterEqual(100, batch["target_ids"].shape[1])  # truncated significantly
            break
