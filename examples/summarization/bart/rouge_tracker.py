from pathlib import Path

import numpy as np
import pandas as pd
import torch

from durbango import lmap, tqdm_nice
from transformers import BartTokenizer


try:
    from finetune import calculate_rouge
except ImportError:
    from .finetune import calculate_rouge


LOGDIR = Path("examples/summarization/bart/dbart/logs/").absolute()
DATA_DIR = Path("examples/summarization/bart/dbart/cnn_dm").absolute()


def rouge_files(src_file:Path, tgt_file:Path):
    src = lmap(str.strip, list(src_file.open().readlines()))
    tgt = lmap(str.strip, list(tgt_file.open().readlines()))
    return calculate_rouge(src, tgt)


def read_gens(exp_name, split="test", n=None):
    expdir = LOGDIR / exp_name
    assert expdir.exists(), expdir
    paths = list(expdir.glob(f"{split}_generations*.txt"))
    assert paths
    path = paths[0]
    lns = lmap(str.strip, list(path.open().readlines()))
    if n is not None:
        return lns[:n]
    return lns


def load_cpu(p):
    return torch.load(p, map_location="cpu")


class RougeTracker:
    def __init__(self, csv_path="rouge_test_df.csv", logdir=LOGDIR, data_dir=DATA_DIR):
        try:
            self.df = pd.read_csv(csv_path, index_col=0)
        except FileNotFoundError:
            self.df = pd.DataFrame()
        self.logdir = logdir
        test_gt = lmap(str.strip, Path(data_dir / "test.target").open().readlines())
        test_gt = lmap(str.strip, test_gt)
        self.gt = test_gt  # {'test': test_gt}
        self.tokenizer = BartTokenizer.from_pretrained("bart-large-cnn")
        self.csv_path = csv_path

    @property
    def finished_experiments(self):
        return [p.parent.name for p in list(self.logdir.glob("*/test_generations*.txt"))]

    def tok_len(self, strang):
        return len(self.tokenizer.encode(strang))

    def read_all_gens(self,):
        GENS = {}
        for f in self.finished_experiments:
            GENS[f] = read_gens(f)
        return GENS

    def score(self, gens, k):
        rouge_raw = calculate_rouge(gens, self.gt)
        lens = np.mean(lmap(self.tok_len, gens))
        return dict(avg_len=lens, exp_name=k, **rouge_raw)

    @property
    def new_experiments(rt):
        return set(rt.finished_experiments).difference(rt.df.index)

    def update(self):
        records = []
        to_score = self.new_experiments
        for exp_name in tqdm_nice(to_score, desc="Rouge Update"):
            gens = read_gens(exp_name)
            if len(gens) != len(self.gt):
                continue
            records.append(self.score(gens, exp_name))
        new_df = (
            pd.DataFrame(records)
            .rename(columns=lambda x: x.replace("rouge", "R"))
            .set_index("exp_name")
            .astype(float)
            .dsort("R2")
        )
        self.df = pd.concat([self.df, new_df])
        return self.df
