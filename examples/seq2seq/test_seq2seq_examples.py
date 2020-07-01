import argparse
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers.testing_utils import require_multigpu

from .distillation import distill_main, evaluate_checkpoint
from .finetune import main
from .run_eval import generate_summaries_or_translations, run_generate
from .utils import SummarizationDataset, lmap, load_json


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
CUDA_AVAILABLE = torch.cuda.is_available()
CHEAP_ARGS = {
    "logger": "default",
    "length_penalty": 0.5,
    "cache_dir": "",
    "task": "summarization",
    "num_workers": 2,
    "alpha_hid": 0,
    "freeze_embeds": True,
    "enc_only": False,
    "tgt_suffix": "",
    "resume_from_checkpoint": None,
    "sortish_sampler": True,
    "student_decoder_layers": 1,
    "val_check_interval": 1.0,
    "output_dir": "",
    "fp16": CUDA_AVAILABLE,
    "no_teacher": False,
    "fp16_opt_level": "O1",
    "gpus": 1 if CUDA_AVAILABLE else 0,
    "n_tpu_cores": 0,
    "max_grad_norm": 1.0,
    "do_train": True,
    "do_predict": True,
    "gradient_accumulation_steps": 1,
    "server_ip": "",
    "server_port": "",
    "seed": 42,
    "model_name_or_path": "sshleifer/bart-tiny-random",
    "config_name": "",
    "tokenizer_name": "facebook/bart-large",
    "do_lower_case": False,
    "learning_rate": 0.3,
    "weight_decay": 0.0,
    "adam_epsilon": 1e-08,
    "warmup_steps": 0,
    "num_train_epochs": 1,
    "train_batch_size": 2,
    "eval_batch_size": 2,
    "max_source_length": 12,
    "max_target_length": 12,
    "val_max_target_length": 12,
    "test_max_target_length": 12,
    "fast_dev_run": False,
    "no_cache": False,
    "n_train": -1,
    "n_val": -1,
    "n_test": -1,
    "student_encoder_layers": 1,
    "alpha_loss_encoder": 0.0,
    "freeze_encoder": False,
    "auto_scale_batch_size": False,
}


def _dump_articles(path: Path, articles: list):
    with path.open("w") as f:
        f.write("\n".join(articles))


ARTICLES = [" Sam ate lunch today", "Sams lunch ingredients"]
SUMMARIES = ["A very interesting story about what I ate for lunch.", "Avocado, celery, turkey, coffee"]
T5_TINY = "patrickvonplaten/t5-tiny-random"
BART_TINY = "sshleifer/bart-tiny-random"
MBART_TINY = "sshleifer/tiny-mbart"
MARIAN_TINY = "sshleifer/tiny-marian-en-de"
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
logging.disable(logging.CRITICAL)  # remove noisy download output from tracebacks


def make_test_data_dir(**kwargs):
    tmp_dir = Path(tempfile.mkdtemp(**kwargs))
    for split in ["train", "val", "test"]:
        _dump_articles((tmp_dir / f"{split}.source"), ARTICLES)
        _dump_articles((tmp_dir / f"{split}.target"), SUMMARIES)
    return tmp_dir


class TestSummarizationDistiller(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)  # remove noisy download output from tracebacks
        return cls

    @require_multigpu
    def test_multigpu(self):
        updates = dict(no_teacher=True, freeze_encoder=True, gpus=2, sortish_sampler=False,)
        self._test_distiller_cli(updates)

    def test_distill_no_teacher(self):
        updates = dict(student_encoder_layers=2, student_decoder_layers=1, no_teacher=True)
        self._test_distiller_cli(updates)

    def test_distill_checkpointing_with_teacher(self):
        updates = dict(
            student_encoder_layers=2,
            student_decoder_layers=1,
            num_train_epochs=4,
            val_check_interval=0.25,
            alpha_hid=2.0,
            model_name_or_path="IGNORE_THIS_IT_DOESNT_GET_USED",
        )
        model = self._test_distiller_cli(updates, check_contents=False)

        ckpts = list(Path(model.output_dir).glob("*.ckpt"))
        self.assertEqual(1, len(ckpts))
        transformer_ckpts = list(Path(model.output_dir).glob("**/*.bin"))
        self.assertEqual(len(transformer_ckpts), 2)
        examples = lmap(str.strip, model.hparams.data_dir.joinpath("test.source").open().readlines())
        out_path = tempfile.mktemp()
        generate_summaries_or_translations(examples, out_path, str(model.output_dir / "best_tfmr"))
        self.assertTrue(Path(out_path).exists())

        evaluate_checkpoint(ckpts[0], dest_dir=Path(tempfile.mkdtemp()))

    @unittest.skip("T5 distillation is broken at the moment")
    def test_distill_t5(self):
        updates = dict(
            student_encoder_layers=1,
            student_decoder_layers=1,
            alpha_hid=2.0,
            teacher=T5_TINY,
            model_name_or_path=T5_TINY,
            tokenizer_name=T5_TINY,
        )
        self._test_distiller_cli(updates)

    def _test_distiller_cli(self, updates, check_contents=True):
        default_updates = dict(
            train_batch_size=1,
            eval_batch_size=2,
            num_train_epochs=2,
            alpha_mlm=0.2,
            alpha_ce=0.8,
            do_predict=True,
            model_name_or_path="sshleifer/tinier_bart",
            teacher=CHEAP_ARGS["model_name_or_path"],
            val_check_interval=0.5,
            alpha_encoder_loss=0.4,
        )
        default_updates.update(updates)
        args_d: dict = CHEAP_ARGS.copy()
        tmp_dir = make_test_data_dir()
        output_dir = tempfile.mkdtemp(prefix="output_")

        args_d.update(data_dir=tmp_dir, output_dir=output_dir, **default_updates)
        model = distill_main(argparse.Namespace(**args_d))
        if not check_contents:
            return model
        contents = os.listdir(output_dir)
        ckpt_name = "val_avg_rouge2=0.0000-step_count=2.ckpt"  # "val_avg_rouge2=0.0000-epoch=1.ckpt"  # "epoch=1-val_avg_rouge2=0.0000.ckpt"
        contents = {os.path.basename(p) for p in contents}
        self.assertIn(ckpt_name, contents)

        self.assertIn("test_generations.txt", contents)
        self.assertIn("test_results.txt", contents)

        metrics = load_json(model.metrics_save_path)
        last_step_stats = metrics["val"][-1]
        self.assertGreaterEqual(last_step_stats["val_avg_gen_time"], 0.01)
        self.assertGreaterEqual(1.0, last_step_stats["val_avg_gen_time"])
        self.assertIsInstance(last_step_stats[f"val_avg_{model.val_metric}"], float)
        desired_n_evals = int(args_d["num_train_epochs"] * (1 / args_d["val_check_interval"]) + 1)
        self.assertEqual(len(metrics["val"]), desired_n_evals)
        self.assertEqual(len(metrics["test"]), 1)
        return model


@pytest.mark.parametrize(["model"], [pytest.param(T5_TINY), pytest.param(BART_TINY), pytest.param(MBART_TINY)])
def test_run_eval_bart(model):
    input_file_name = Path(tempfile.mkdtemp()) / "utest_input.source"
    output_file_name = input_file_name.parent / "utest_output.txt"
    assert not output_file_name.exists()
    articles = [" New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County."]
    _dump_articles(input_file_name, articles)
    testargs = ["run_eval.py", model, str(input_file_name), str(output_file_name)]  # TODO: test score_path
    with patch.object(sys, "argv", testargs):
        run_generate()
        assert Path(output_file_name).exists()
        os.remove(Path(output_file_name))


@pytest.mark.parametrize(
    ["model"], [pytest.param(T5_TINY), pytest.param(BART_TINY), pytest.param(MBART_TINY), pytest.param(MARIAN_TINY)]
)
def test_finetune(model):
    args_d: dict = CHEAP_ARGS.copy()
    task = "translation" if model in [MBART_TINY, MARIAN_TINY] else "summarization"
    tmp_dir = make_test_data_dir()
    output_dir = tempfile.mkdtemp(prefix="output_")
    args_d.update(
        data_dir=tmp_dir,
        model_name_or_path=model,
        tokenizer_name=None,
        train_batch_size=2,
        eval_batch_size=2,
        output_dir=output_dir,
        do_predict=True,
        task=task,
    )
    assert "n_train" in args_d
    args = argparse.Namespace(**args_d)
    main(args)


@pytest.mark.parametrize(
    ["tok"], [pytest.param(T5_TINY), pytest.param(BART_TINY), pytest.param(MBART_TINY), pytest.param(MARIAN_TINY)]
)
def test_dataset(tok):
    tokenizer = AutoTokenizer.from_pretrained(tok)
    tmp_dir = make_test_data_dir()
    max_len_source = max(len(tokenizer.encode(a)) for a in ARTICLES)
    max_len_target = max(len(tokenizer.encode(a)) for a in SUMMARIES)
    trunc_target = 4
    train_dataset = SummarizationDataset(
        tokenizer, data_dir=tmp_dir, type_path="train", max_source_length=20, max_target_length=trunc_target,
    )
    dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=train_dataset.collate_fn)
    for batch in dataloader:
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        # show that articles were trimmed.
        assert batch["input_ids"].shape[1] == max_len_source
        assert 20 >= batch["input_ids"].shape[1]  # trimmed significantly
        # show that targets were truncated
        assert batch["decoder_input_ids"].shape[1] == trunc_target  # Truncated
        assert max_len_target > trunc_target  # Truncated
