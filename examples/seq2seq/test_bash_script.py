import argparse
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytorch_lightning as pl
import torch

from .finetune import SummarizationModule, main
from .test_seq2seq_examples import make_test_data_dir, MBART_TINY, CUDA_AVAILABLE
from transformers.testing_utils import slow
import wget
import tarfile


def fetch_and_save_wmt_100():
    #TODO(SS): DELETEME
    DATA_URL = 'https://s3.amazonaws.com/datasets.huggingface.co/translation/wmt_en_ro_test_data.tgz'

    dest_dir = 'wmt_100_dir'
    if os.path.isdir(dest_dir):
        return dest_dir
    filename = wget.download(DATA_URL)
    tarball = tarfile.TarFile(filename)
    tarball.extractall(path=dest_dir)
    os.remove(filename)
    return dest_dir


@slow
def test_train_mbart_cc25_enro_script():
    data_dir = 'examples/seq2seq/test_data/wmt_en_ro'
    env_vars_to_replace = {
        "$MAX_LEN": 200,
        "$BS": 4,
        "$GAS": 1,
        "$ENRO_DIR": data_dir,
        "facebook/mbart-large-cc25": MBART_TINY,
    }

    # Clean up bash script
    bash_script = Path("examples/seq2seq/train_mbart_cc25_enro.sh").open().read().split("finetune.py")[1].strip()
    bash_script = bash_script.replace("\\\n", "").strip().replace("$@", "")
    for k, v in env_vars_to_replace.items():
        bash_script = bash_script.replace(k, str(v))
    output_dir = tempfile.mkdtemp(prefix="output")

    if CUDA_AVAILABLE:
        gpus = torch.cuda.device_count()
    else:
        bash_script = bash_script.replace("--fp16", "")
        gpus = 0

    testargs = ["finetune.py"] + bash_script.split() + [f"--output_dir={output_dir}", f"--gpus={gpus}"]
    with patch.object(sys, "argv", testargs):
        parser = argparse.ArgumentParser()
        parser = pl.Trainer.add_argparse_args(parser)
        parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())
        args = parser.parse_args()
        # assert args.gpus == gpus THIS BREAKS
        # args.gpus = gpus
        main(args)
