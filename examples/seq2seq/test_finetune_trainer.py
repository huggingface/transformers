import glob
import os
import sys
import tempfile
from unittest.mock import patch

import pytest
import timeout_decorator

from transformers import BartForConditionalGeneration, MarianMTModel
from transformers.testing_utils import slow

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
    MarianMTModel.from_pretrained(MARIAN_MODEL)


@timeout_decorator.timeout(600)
@slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="too slow to run on CPU")
def test_finetune_trainer():
    data_dir = "examples/seq2seq/test_data/wmt_en_ro"
    output_dir = tempfile.mkdtemp(prefix="marian_output")
    max_len = "56"
    num_train_epochs = 6
    eval_steps = 2
    argv = [
        "--model_name_or_path",
        MARIAN_MODEL,
        "--data_dir",
        data_dir,
        "--output_dir",
        output_dir,
        "--overwrite_output_dir",
        "--n_train",
        "8",
        "--n_val",
        "8",
        "--max_source_length",
        max_len,
        "--max_target_length",
        max_len,
        "--val_max_target_length",
        max_len,
        "--do_train",
        "--do_eval",
        "--num_train_epochs",
        str(num_train_epochs),
        "--per_device_train_batch_size",
        "4",
        "--per_device_eval_batch_size",
        "4",
        "--learning_rate",
        "1e-3",
        "--warmup_steps",
        "10",
        "--evaluate_during_training",
        "--predict_with_generate",
        "--logging_steps",
        str(eval_steps),
        "--save_steps",
        str(eval_steps),
        "--eval_steps",
        str(eval_steps),
        "--sortish_sampler",
        "--label_smoothing",
        "0.1",
    ]

    testargs = ["finetune_trainer.py"] + argv
    with patch.object(sys, "argv", testargs):
        main()

    # TODO: check that saved files work

    # check checkpoint dirs
    ckpt_dirs = glob.glob(f"{output_dir}/checkpoint*")
    num_expected_ckpt = 6
    assert len(ckpt_dirs) == num_expected_ckpt

    # Check metrics
    first_metrics_save_path = os.path.join(ckpt_dirs[0], "eval_results.json")
    first_step_stats = load_json(first_metrics_save_path)
    last_metrics_save_path = os.path.join(ckpt_dirs[-1], "eval_results.json")
    last_step_stats = load_json(last_metrics_save_path)

    assert last_step_stats["val_avg_gen_time"] >= 0.01

    assert first_step_stats["val_avg_bleu"] < last_step_stats["val_avg_bleu"]  # model learned nothing
    assert 1.0 >= last_step_stats["val_avg_gen_time"]  # model hanging on generate. Maybe bad config was saved.
    assert isinstance(last_step_stats["val_avg_bleu"], float)

    # TODO(SS): turn on args.do_predict when PL bug fixed.
    # if args.do_predict:
    #     contents = {os.path.basename(p) for p in contents}
    #     assert "test_generations.txt" in contents
    #     assert "test_results.txt" in contents
    #     # assert len(metrics["val"]) ==  desired_n_evals
    #     assert len(metrics["test"]) == 1
