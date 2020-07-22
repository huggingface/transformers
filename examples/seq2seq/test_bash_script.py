import argparse
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import pytorch_lightning as pl
import timeout_decorator
import torch
import wget

from transformers.testing_utils import slow
from transformers import BartForConditionalGeneration

from .finetune import SummarizationModule, main
from .test_seq2seq_examples import CUDA_AVAILABLE
from .utils import load_json


def fetch_and_save_wmt_100():
    # TODO(SS): DELETEME
    DATA_URL = "https://s3.amazonaws.com/datasets.huggingface.co/translation/wmt_en_ro_test_data.tgz"

    dest_dir = "wmt_100_dir"
    if os.path.isdir(dest_dir):
        return dest_dir
    filename = wget.download(DATA_URL)
    tarball = tarfile.TarFile(filename)
    tarball.extractall(path=dest_dir)
    os.remove(filename)
    return dest_dir

MODEL_NAME = "sshleifer/student_mbart_en_ro_1_1"
@slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="too slow to run on CPU")
def test_student_download():
    """This warms up the cache so that we can time the next test without including download time."""
    BartForConditionalGeneration.from_pretrained(MODEL_NAME)



@timeout_decorator.timeout(60)
@slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="too slow to run on CPU")
def test_train_mbart_cc25_enro_script():
    data_dir = "examples/seq2seq/test_data/wmt_en_ro"
    env_vars_to_replace = {
        "$MAX_LEN": 200,
        "$BS": 4,
        "$GAS": 1,
        "$ENRO_DIR": data_dir,
        "facebook/mbart-large-cc25": MODEL_NAME,
        # 1 encoder and 1 decoder layer from finetuned mbart en-ro. Should be able to start >0 and improve quickly.
        # Download is 600MB in previous test.
        "val_check_interval=0.25": "val_check_interval=1.0",
    }

    # Clean up bash script
    bash_script = Path("examples/seq2seq/train_mbart_cc25_enro.sh").open().read().split("finetune.py")[1].strip()
    bash_script = bash_script.replace("\\\n", "").strip().replace("$@", "")
    for k, v in env_vars_to_replace.items():
        bash_script = bash_script.replace(k, str(v))
    output_dir = tempfile.mkdtemp(prefix="output")

    if CUDA_AVAILABLE:
        gpus = 1  # torch.cuda.device_count()
    else:
        bash_script = bash_script.replace("--fp16", "")
        gpus = 0

    testargs = (
        ["finetune.py"]
        + bash_script.split()
        + [
            f"--output_dir={output_dir}",
            f"--gpus={gpus}",
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
    contents = os.listdir(output_dir)

    ckpt_path = [x for x in contents if x.endswith(".ckpt")][0]
    full_path = os.path.join(args.output_dir, ckpt_path)
    ckpt = torch.load(full_path, map_location="cpu")
    assert ckpt["global_step"] == (args.max_epochs / args.val_check_interval) + 1  # extra step for sanity check

    metrics = load_json(model.metrics_save_path)
    first_step_stats = metrics["val"][0]
    last_step_stats = metrics["val"][-1]
    assert last_step_stats["val_avg_gen_time"] >= 0.01
    assert 1.0 >= last_step_stats["val_avg_gen_time"]
    assert first_step_stats["val_avg_bleu"] < last_step_stats["val_avg_bleu"]
    # TODO(SS): check that val ran every val_check_interval
    # test that takes less than 70
    assert isinstance(last_step_stats[f"val_avg_{model.val_metric}"], float)
    # desired_n_evals = int(args_d["max_epochs"] * (1 / args_d["val_check_interval"]) + 1)
    if args.do_predict:
        contents = {os.path.basename(p) for p in contents}
        assert "test_generations.txt" in contents
        assert "test_results.txt" in contents
        # assert len(metrics["val"]) ==  desired_n_evals
        assert len(metrics["test"]) == 1
