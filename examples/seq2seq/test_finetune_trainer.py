import os
import sys
import tempfile
from unittest.mock import patch

from transformers.testing_utils import slow
from transformers.trainer_utils import set_seed

from .finetune_trainer import main
from .test_seq2seq_examples import MBART_TINY
from .utils import load_json


set_seed(42)
MARIAN_MODEL = "sshleifer/student_marian_en_ro_6_1"


def test_finetune_trainer():
    output_dir = run_trainer(1, "12", MBART_TINY, 1)
    logs = load_json(os.path.join(output_dir, "log_history.json"))
    eval_metrics = [log for log in logs if "eval_loss" in log.keys()]
    first_step_stats = eval_metrics[0]
    assert "eval_bleu" in first_step_stats


@slow
def test_finetune_trainer_slow():
    # TODO(SS): This will fail on devices with more than 1 GPU.
    # There is a missing call to __init__process_group somewhere
    output_dir = run_trainer(eval_steps=2, max_len="32", model_name=MARIAN_MODEL, num_train_epochs=3)

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


def run_trainer(eval_steps: int, max_len: str, model_name: str, num_train_epochs: int):
    data_dir = "examples/seq2seq/test_data/wmt_en_ro"
    output_dir = tempfile.mkdtemp(prefix="test_output")
    argv = [
        "--model_name_or_path",
        model_name,
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
        # "--eval_beams",
        # "2",
        "--task",
        "translation",
        "--tgt_lang",
        "ro_RO",
        "--src_lang",
        "en_XX",
    ]
    testargs = ["finetune_trainer.py"] + argv
    with patch.object(sys, "argv", testargs):
        main()

    return output_dir
