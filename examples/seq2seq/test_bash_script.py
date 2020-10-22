#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import pytorch_lightning as pl
import timeout_decorator
import torch

from distillation import BartSummarizationDistiller, distill_main
from finetune import SummarizationModule, main
from test_seq2seq_examples import CUDA_AVAILABLE, MBART_TINY
from transformers import BartForConditionalGeneration, MarianMTModel
from transformers.testing_utils import TestCasePlus, slow
from utils import load_json


MODEL_NAME = MBART_TINY
MARIAN_MODEL = "sshleifer/student_marian_en_ro_6_1"


class TestAll(TestCasePlus):
    @slow
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="too slow to run on CPU")
    def test_model_download(self):
        """This warms up the cache so that we can time the next test without including download time, which varies between machines."""
        BartForConditionalGeneration.from_pretrained(MODEL_NAME)
        MarianMTModel.from_pretrained(MARIAN_MODEL)

    @timeout_decorator.timeout(120)
    @slow
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="too slow to run on CPU")
    def test_train_mbart_cc25_enro_script(self):
        data_dir = "examples/seq2seq/test_data/wmt_en_ro"
        env_vars_to_replace = {
            "--fp16_opt_level=O1": "",
            "$MAX_LEN": 128,
            "$BS": 4,
            "$GAS": 1,
            "$ENRO_DIR": data_dir,
            "facebook/mbart-large-cc25": MODEL_NAME,
            # Download is 120MB in previous test.
            "val_check_interval=0.25": "val_check_interval=1.0",
        }

        # Clean up bash script
        bash_script = Path("examples/seq2seq/train_mbart_cc25_enro.sh").open().read().split("finetune.py")[1].strip()
        bash_script = bash_script.replace("\\\n", "").strip().replace('"$@"', "")
        for k, v in env_vars_to_replace.items():
            bash_script = bash_script.replace(k, str(v))
        output_dir = self.get_auto_remove_tmp_dir()

        bash_script = bash_script.replace("--fp16 ", "")
        testargs = (
            ["finetune.py"]
            + bash_script.split()
            + [
                f"--output_dir={output_dir}",
                "--gpus=1",
                "--learning_rate=3e-1",
                "--warmup_steps=0",
                "--val_check_interval=1.0",
                "--tokenizer_name=facebook/mbart-large-en-ro",
            ]
        )
        with patch.object(sys, "argv", testargs):
            parser = argparse.ArgumentParser()
            parser = pl.Trainer.add_argparse_args(parser)
            parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())
            args = parser.parse_args()
            args.do_predict = False
            # assert args.gpus == gpus THIS BREAKS for multigpu
            model = main(args)

        # Check metrics
        metrics = load_json(model.metrics_save_path)
        first_step_stats = metrics["val"][0]
        last_step_stats = metrics["val"][-1]
        assert (
            len(metrics["val"]) == (args.max_epochs / args.val_check_interval) + 1
        )  # +1 accounts for val_sanity_check

        assert last_step_stats["val_avg_gen_time"] >= 0.01

        assert first_step_stats["val_avg_bleu"] < last_step_stats["val_avg_bleu"]  # model learned nothing
        assert 1.0 >= last_step_stats["val_avg_gen_time"]  # model hanging on generate. Maybe bad config was saved.
        assert isinstance(last_step_stats[f"val_avg_{model.val_metric}"], float)

        # check lightning ckpt can be loaded and has a reasonable statedict
        contents = os.listdir(output_dir)
        ckpt_path = [x for x in contents if x.endswith(".ckpt")][0]
        full_path = os.path.join(args.output_dir, ckpt_path)
        ckpt = torch.load(full_path, map_location="cpu")
        expected_key = "model.model.decoder.layers.0.encoder_attn_layer_norm.weight"
        assert expected_key in ckpt["state_dict"]
        assert ckpt["state_dict"]["model.model.decoder.layers.0.encoder_attn_layer_norm.weight"].dtype == torch.float32

        # TODO: turn on args.do_predict when PL bug fixed.
        if args.do_predict:
            contents = {os.path.basename(p) for p in contents}
            assert "test_generations.txt" in contents
            assert "test_results.txt" in contents
            # assert len(metrics["val"]) ==  desired_n_evals
            assert len(metrics["test"]) == 1

    @timeout_decorator.timeout(600)
    @slow
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="too slow to run on CPU")
    def test_opus_mt_distill_script(self):
        data_dir = "examples/seq2seq/test_data/wmt_en_ro"
        env_vars_to_replace = {
            "--fp16_opt_level=O1": "",
            "$MAX_LEN": 128,
            "$BS": 16,
            "$GAS": 1,
            "$ENRO_DIR": data_dir,
            "$m": "sshleifer/student_marian_en_ro_6_1",
            "val_check_interval=0.25": "val_check_interval=1.0",
        }

        # Clean up bash script
        bash_script = (
            Path("examples/seq2seq/distil_marian_no_teacher.sh").open().read().split("distillation.py")[1].strip()
        )
        bash_script = bash_script.replace("\\\n", "").strip().replace('"$@"', "")
        bash_script = bash_script.replace("--fp16 ", " ")

        for k, v in env_vars_to_replace.items():
            bash_script = bash_script.replace(k, str(v))
        output_dir = self.get_auto_remove_tmp_dir()
        bash_script = bash_script.replace("--fp16", "")
        epochs = 6
        testargs = (
            ["distillation.py"]
            + bash_script.split()
            + [
                f"--output_dir={output_dir}",
                "--gpus=1",
                "--learning_rate=1e-3",
                f"--num_train_epochs={epochs}",
                "--warmup_steps=10",
                "--val_check_interval=1.0",
            ]
        )
        with patch.object(sys, "argv", testargs):
            parser = argparse.ArgumentParser()
            parser = pl.Trainer.add_argparse_args(parser)
            parser = BartSummarizationDistiller.add_model_specific_args(parser, os.getcwd())
            args = parser.parse_args()
            args.do_predict = False
            # assert args.gpus == gpus THIS BREAKS for multigpu

            model = distill_main(args)

        # Check metrics
        metrics = load_json(model.metrics_save_path)
        first_step_stats = metrics["val"][0]
        last_step_stats = metrics["val"][-1]
        assert len(metrics["val"]) >= (args.max_epochs / args.val_check_interval)  # +1 accounts for val_sanity_check

        assert last_step_stats["val_avg_gen_time"] >= 0.01

        assert first_step_stats["val_avg_bleu"] < last_step_stats["val_avg_bleu"]  # model learned nothing
        assert 1.0 >= last_step_stats["val_avg_gen_time"]  # model hanging on generate. Maybe bad config was saved.
        assert isinstance(last_step_stats[f"val_avg_{model.val_metric}"], float)

        # check lightning ckpt can be loaded and has a reasonable statedict
        contents = os.listdir(output_dir)
        ckpt_path = [x for x in contents if x.endswith(".ckpt")][0]
        full_path = os.path.join(args.output_dir, ckpt_path)
        ckpt = torch.load(full_path, map_location="cpu")
        expected_key = "model.model.decoder.layers.0.encoder_attn_layer_norm.weight"
        assert expected_key in ckpt["state_dict"]
        assert ckpt["state_dict"]["model.model.decoder.layers.0.encoder_attn_layer_norm.weight"].dtype == torch.float32

        # TODO: turn on args.do_predict when PL bug fixed.
        if args.do_predict:
            contents = {os.path.basename(p) for p in contents}
            assert "test_generations.txt" in contents
            assert "test_results.txt" in contents
            # assert len(metrics["val"]) ==  desired_n_evals
            assert len(metrics["test"]) == 1
