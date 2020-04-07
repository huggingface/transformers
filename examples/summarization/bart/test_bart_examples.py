import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from torch.utils.data import DataLoader

from transformers import BartTokenizer

from .evaluate_cnn import run_generate
from .utils import SummarizationDataset


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def _dump_articles(path: Path, articles: list):
    with path.open("w") as f:
        f.write("\n".join(articles))


class TestBartExamples(unittest.TestCase):
    def test_bart_cnn_cli(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        tmp = Path(tempfile.gettempdir()) / "utest_generations_bart_sum.hypo"
        output_file_name = Path(tempfile.gettempdir()) / "utest_output_bart_sum.hypo"
        articles = [" New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County."]
        _dump_articles(tmp, articles)
        testargs = ["evaluate_cnn.py", str(tmp), str(output_file_name), "sshleifer/bart-tiny-random"]
        with patch.object(sys, "argv", testargs):
            run_generate()
            self.assertTrue(Path(output_file_name).exists())

    def test_bart_summarization_dataset(self):
        tmp_dir = Path(tempfile.gettempdir())
        articles = [" Sam ate lunch today", "Sams lunch ingredients"]
        summaries = ["A very interesting story about what I ate for lunch.", "Avocado, celery, turkey, coffee"]
        _dump_articles((tmp_dir / "train.source"), articles)
        _dump_articles((tmp_dir / "train.target"), summaries)
        tokenizer = BartTokenizer.from_pretrained("bart-large")
        max_len_source = max(len(tokenizer.encode(a)) for a in articles)
        max_len_target = max(len(tokenizer.encode(a)) for a in summaries)
        trunc_target = 4
        train_dataset = SummarizationDataset(
            tokenizer, data_dir=tmp_dir, type_path="train", max_source_length=20, max_target_length=trunc_target,
        )
        dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=train_dataset.collate_fn)
        for batch in dataloader:
            self.assertEqual(batch["source_mask"].shape, batch["source_ids"].shape)
            # show that articles were trimmed.
            self.assertEqual(batch["source_ids"].shape[1], max_len_source)
            self.assertGreater(20, batch["source_ids"].shape[1])  # trimmed significantly

            # show that targets were truncated
            self.assertEqual(batch["target_ids"].shape[1], trunc_target)  # Truncated
            self.assertGreater(max_len_target, trunc_target)  # Truncated
