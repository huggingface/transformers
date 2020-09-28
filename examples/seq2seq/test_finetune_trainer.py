import os
import sys
import tempfile
from unittest.mock import patch

from transformers import BartForConditionalGeneration, MarianMTModel
from transformers.testing_utils import slow

from .finetune_trainer import main
from .test_seq2seq_examples import MBART_TINY
from .utils import load_json


MODEL_NAME = MBART_TINY
# TODO(SS): MODEL_NAME = "sshleifer/student_mbart_en_ro_1_1"
MARIAN_MODEL = "sshleifer/student_marian_en_ro_6_1"


@slow
def test_model_download():
    """This warms up the cache so that we can time the next test without including download time, which varies between machines."""
    BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    MarianMTModel.from_pretrained(MARIAN_MODEL)


@slow
def test_finetune_trainer():
    data_dir = "examples/seq2seq/test_data/wmt_en_ro"
    output_dir = tempfile.mkdtemp(prefix="marian_output")
    max_len = "128"
    num_train_epochs = 4
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
        "--do_predict",
        "--num_train_epochs",
        str(num_train_epochs),
        "--per_device_train_batch_size",
        "4",
        "--per_device_eval_batch_size",
        "4",
        "--learning_rate",
        "3e-4",
        "--warmup_steps",
        "8",
        "--evaluate_during_training",
        "--predict_with_generate",
        "--logging_steps",
        0,
        "--save_steps",
        str(eval_steps),
        "--eval_steps",
        str(eval_steps),
        "--sortish_sampler",
        "--label_smoothing",
        "0.1",
        "--task",
        "translation",
    ]

    testargs = ["finetune_trainer.py"] + argv
    with patch.object(sys, "argv", testargs):
        main()

    # Check metrics
    logs = load_json(os.path.join(output_dir, "log_history.json"))
    eval_metrics = [log for log in logs if "eval_loss" in log.keys()]
    first_step_stats = eval_metrics[0]
    last_step_stats = eval_metrics[-1]

    assert first_step_stats["eval_bleu"] < last_step_stats["eval_bleu"]  # model learned nothing
    assert isinstance(last_step_stats["eval_bleu"], float)

    # test if do_predict saves generations and metrics
    contents = os.listdir(output_dir)
    contents = {os.path.basename(p) for p in contents}
    assert "test_generations.txt" in contents
    assert "test_results.json" in contents
