#!/usr/bin/env python

import argparse
import os
import sys
from unittest.mock import patch

import pytorch_lightning as pl
import timeout_decorator
import torch

from distillation import SummarizationDistiller, distill_main
from finetune import SummarizationModule, main
from transformers import MarianMTModel
from transformers.file_utils import cached_path
from transformers.testing_utils import TestCasePlus, require_torch_gpu, slow
from utils import load_json


MARIAN_MODEL = "sshleifer/mar_enro_6_3_student"


class TestMbartCc25Enro(TestCasePlus):
    def setUp(self):
        super().setUp()

        data_cached = cached_path(
            "https://cdn-datasets.huggingface.co/translation/wmt_en_ro-tr40k-va0.5k-te0.5k.tar.gz",
            extract_compressed_file=True,
        )
        self.data_dir = f"{data_cached}/wmt_en_ro-tr40k-va0.5k-te0.5k"

    @slow
    @require_torch_gpu
    def test_model_download(self):
        """This warms up the cache so that we can time the next test without including download time, which varies between machines."""
        MarianMTModel.from_pretrained(MARIAN_MODEL)

    # @timeout_decorator.timeout(1200)
    @slow
    @require_torch_gpu
    def test_train_mbart_cc25_enro_script(self):
        env_vars_to_replace = {
            "$MAX_LEN": 64,
            "$BS": 64,
            "$GAS": 1,
            "$ENRO_DIR": self.data_dir,
            "facebook/mbart-large-cc25": MARIAN_MODEL,
            # "val_check_interval=0.25": "val_check_interval=1.0",
            "--learning_rate=3e-5": "--learning_rate 3e-4",
            "--num_train_epochs 6": "--num_train_epochs 1",
        }

        # Clean up bash script
        bash_script = (self.test_file_dir / "train_mbart_cc25_enro.sh").open().read().split("finetune.py")[1].strip()
        bash_script = bash_script.replace("\\\n", "").strip().replace('"$@"', "")
        for k, v in env_vars_to_replace.items():
            bash_script = bash_script.replace(k, str(v))
        output_dir = self.get_auto_remove_tmp_dir()

        # bash_script = bash_script.replace("--fp16 ", "")
        args = f"""
            --output_dir {output_dir}
            --tokenizer_name Helsinki-NLP/opus-mt-en-ro
            --sortish_sampler
            --do_predict
            --gpus 1
            --freeze_encoder
            --n_train 40000
            --n_val 500
            --n_test 500
            --fp16_opt_level O1
            --num_sanity_val_steps 0
            --eval_beams 2
        """.split()
        # XXX: args.gpus > 1 : handle multi_gpu in the future

        testargs = ["finetune.py"] + bash_script.split() + args
        with patch.object(sys, "argv", testargs):
            parser = argparse.ArgumentParser()
            parser = pl.Trainer.add_argparse_args(parser)
            parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())
            args = parser.parse_args()
            model = main(args)

        # Check metrics
        metrics = load_json(model.metrics_save_path)
        first_step_stats = metrics["val"][0]
        last_step_stats = metrics["val"][-1]
        self.assertEqual(len(metrics["val"]), (args.max_epochs / args.val_check_interval))
        assert isinstance(last_step_stats[f"val_avg_{model.val_metric}"], float)

        self.assertGreater(last_step_stats["val_avg_gen_time"], 0.01)
        # model hanging on generate. Maybe bad config was saved. (XXX: old comment/assert?)
        self.assertLessEqual(last_step_stats["val_avg_gen_time"], 1.0)

        # test learning requirements:

        # 1. BLEU improves over the course of training by more than 2 pts
        self.assertGreater(last_step_stats["val_avg_bleu"] - first_step_stats["val_avg_bleu"], 2)

        # 2. BLEU finishes above 17
        self.assertGreater(last_step_stats["val_avg_bleu"], 17)

        # 3. test BLEU and val BLEU within ~1.1 pt.
        self.assertLess(abs(metrics["val"][-1]["val_avg_bleu"] - metrics["test"][-1]["test_avg_bleu"]), 1.1)

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


class TestDistilMarianNoTeacher(TestCasePlus):
    @timeout_decorator.timeout(600)
    @slow
    @require_torch_gpu
    def test_opus_mt_distill_script(self):
        data_dir = f"{self.test_file_dir_str}/test_data/wmt_en_ro"
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
            (self.test_file_dir / "distil_marian_no_teacher.sh").open().read().split("distillation.py")[1].strip()
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
                "--do_predict",
            ]
        )
        with patch.object(sys, "argv", testargs):
            parser = argparse.ArgumentParser()
            parser = pl.Trainer.add_argparse_args(parser)
            parser = SummarizationDistiller.add_model_specific_args(parser, os.getcwd())
            args = parser.parse_args()
            # assert args.gpus == gpus THIS BREAKS for multi_gpu

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
