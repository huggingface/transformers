import argparse
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import pytorch_lightning as pl
import timeout_decorator
import torch

from transformers import BartForConditionalGeneration, MarianMTModel
from transformers.testing_utils import slow

from .distillation import BartSummarizationDistiller, distill_main
from .finetune import SummarizationModule, main
from .finetune_trainer import main
from .test_seq2seq_examples import CUDA_AVAILABLE, MBART_TINY
from .utils import load_json


MODEL_NAME = MBART_TINY
# TODO(SS): MODEL_NAME = "sshleifer/student_mbart_en_ro_1_1"
MARIAN_MODEL = "sshleifer/student_marian_en_ro_6_1"


@slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="too slow to run on CPU")
def test_model_download():
    """This warms up the cache so that we can time the next test without including download time, which varies between machines."""
    BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    # MarianMTModel.from_pretrained(MARIAN_MODEL)


@timeout_decorator.timeout(600)
# @slow
# @pytest.mark.skipif(not CUDA_AVAILABLE, reason="too slow to run on CPU")
def test_finetune_trainer():
    data_dir = "examples/seq2seq/test_data/wmt_en_ro"
    max_len = "56"
    argv = [
        "--model_name_or_path",
        MARIAN_MODEL,
        "--data_dir",
        data_dir,
        "--overwrite_output_dir",
        # '--n_train', '8', '--n_val', '8',
        "--max_source_length",
        max_len,
        "--max_target_length",
        max_len,
        "--val_max_target_length",
        max_len,
        "--do_train",
        "--do_eval",
        "--num_train_epochs",
        "2",
        "--per_device_train_batch_size",
        "4",
        "--per_device_eval_batch_size",
        "4",
        "--evaluate_during_training",
        "--predict_with_generate",
        "--logging_steps",
        "2",
        "--save_steps",
        "2",
        "--eval_steps",
        "2",
        "--sortish_sampler",
        "--label_smoothing",
        "0.1",
    ]
    output_dir = tempfile.mkdtemp(prefix="marian_output")
    epochs = 6
    testargs = (
        ["finetune_trainer.py"]
        + argv
        + [
            "--output_dir=test",
            # f"--output_dir={output_dir}",
            # "--gpus=1",
            # "--learning_rate=1e-3",
            # f"--num_train_epochs={epochs}",
            # "--warmup_steps=10",
            # "--val_check_interval=1.0",
        ]
    )
    with patch.object(sys, "argv", testargs):
        main()
    #TODO: check that saved files work
    return

    # Check metrics
    metrics = load_json(model.metrics_save_path)
    first_step_stats = metrics["val"][0]
    last_step_stats = metrics["val"][-1]
    assert len(metrics["val"]) == (args.max_epochs / args.val_check_interval) + 1  # +1 accounts for val_sanity_check

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

    # TODO(SS): turn on args.do_predict when PL bug fixed.
    if args.do_predict:
        contents = {os.path.basename(p) for p in contents}
        assert "test_generations.txt" in contents
        assert "test_results.txt" in contents
        # assert len(metrics["val"]) ==  desired_n_evals
        assert len(metrics["test"]) == 1
