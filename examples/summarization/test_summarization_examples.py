import argparse
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from torch.utils.data import DataLoader

from transformers import BartTokenizer

from .evaluate_cnn import run_generate
from .finetune import main
from .utils import SummarizationDataset


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
    "max_source_length": 12,
    "max_target_length": 12,
}


def _dump_articles(path: Path, articles: list):
    with path.open("w") as f:
        f.write("\n".join(articles))


def make_test_data_dir():
    tmp_dir = Path(tempfile.gettempdir())
    articles = [" Sam ate lunch today", "Sams lunch ingredients"]
    summaries = ["A very interesting story about what I ate for lunch.", "Avocado, celery, turkey, coffee"]
    for split in ["train", "val", "test"]:
        _dump_articles((tmp_dir / f"{split}.source"), articles)
        _dump_articles((tmp_dir / f"{split}.target"), summaries)
    return tmp_dir


class TestBartExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        logging.disable(logging.CRITICAL)  # remove noisy download output from tracebacks
        return cls

    def test_bart_cnn_cli(self):
        tmp = Path(tempfile.gettempdir()) / "utest_generations_bart_sum.hypo"
        output_file_name = Path(tempfile.gettempdir()) / "utest_output_bart_sum.hypo"
        articles = [" New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County."]
        _dump_articles(tmp, articles)
        testargs = ["evaluate_cnn.py", str(tmp), str(output_file_name), "sshleifer/bart-tiny-random"]
        with patch.object(sys, "argv", testargs):
            run_generate()
            self.assertTrue(Path(output_file_name).exists())
            os.remove(Path(output_file_name))

    def test_bart_run_sum_cli(self):
        args_d: dict = DEFAULT_ARGS.copy()
        tmp_dir = make_test_data_dir()
        output_dir = tempfile.mkdtemp(prefix="output_")
        args_d.update(
            data_dir=tmp_dir, model_type="bart", train_batch_size=2, eval_batch_size=2, n_gpu=0, output_dir=output_dir,
        )
        main(argparse.Namespace(**args_d))
        args_d.update({"do_train": False, "do_predict": True})

        main(argparse.Namespace(**args_d))
        contents = os.listdir(output_dir)
        expected_contents = {
            "checkpointepoch=0.ckpt",
            "test_results.txt",
        }
        created_files = {os.path.basename(p) for p in contents}
        self.assertSetEqual(expected_contents, created_files)

    def test_t5_run_sum_cli(self):
        args_d: dict = DEFAULT_ARGS.copy()
        tmp_dir = make_test_data_dir()
        output_dir = tempfile.mkdtemp(prefix="output_")
        args_d.update(
            data_dir=tmp_dir,
            model_type="t5",
            model_name_or_path="patrickvonplaten/t5-tiny-random",
            train_batch_size=2,
            eval_batch_size=2,
            n_gpu=0,
            output_dir=output_dir,
            do_predict=True,
        )
        main(argparse.Namespace(**args_d))

        # args_d.update({"do_train": False, "do_predict": True})
        # main(argparse.Namespace(**args_d))

    def test_bart_summarization_dataset(self):
        tmp_dir = Path(tempfile.gettempdir())
        articles = [" Sam ate lunch today", "Sams lunch ingredients"]
        summaries = ["A very interesting story about what I ate for lunch.", "Avocado, celery, turkey, coffee"]
        _dump_articles((tmp_dir / "train.source"), articles)
        _dump_articles((tmp_dir / "train.target"), summaries)
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
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


class TestT5Examples(unittest.TestCase):
    def test_t5_cli(self):
        output_file_name = "output_t5_sum.txt"
        score_file_name = "score_t5_sum.txt"
        articles = ["New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County."]
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        tmp = Path(tempfile.gettempdir()) / "utest_generations_t5_sum.hypo"
        with tmp.open("w", encoding="utf-8") as f:
            f.write("\n".join(articles))

        output_file_name = Path(tempfile.gettempdir()) / "utest_output_t5_sum.hypo"
        score_file_name = Path(tempfile.gettempdir()) / "utest_score_t5_sum.hypo"

        testargs = [
            "evaluate_cnn.py",
            str(tmp),
            str(output_file_name),
            "patrickvonplaten/t5-tiny-random",
            "--reference_path",
            str(tmp),
            "--score_path",
            str(score_file_name),
        ]

        with patch.object(sys, "argv", testargs):
            run_generate()
            self.assertTrue(Path(output_file_name).exists())
            self.assertTrue(Path(score_file_name).exists())
