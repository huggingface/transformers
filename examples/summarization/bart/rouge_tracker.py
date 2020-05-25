from pathlib import Path

import numpy as np
import pandas as pd
import torch

from durbango import lmap
from transformers import BartTokenizer


try:
    from finetune import calculate_rouge
except ImportError:
    from .finetune import calculate_rouge


LOGDIR = Path("examples/summarization/bart/dbart/logs/").absolute()
DATA_DIR = Path("examples/summarization/bart/dbart/cnn_dm").absolute()


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



from durbango import tqdm_nice
def load_cpu(p):
    return torch.load(p, map_location="cpu")


class RougeTracker:
    def __init__(self, csv_path="rouge_test_df.csv", logdir=LOGDIR, data_dir=DATA_DIR):
        self.df = pd.read_csv(csv_path, index_col=0)
        self.rouged_experiments = self.df.index
        self.finished_experiments = [p.parent.name for p in list(logdir.glob("*/test_generations*.txt"))]
        self.logdir = LOGDIR
        test_gt = lmap(str.strip, Path(data_dir / "test.target").open().readlines())
        test_gt = lmap(str.strip, test_gt)
        self.gt = test_gt  # {'test': test_gt}
        self.tokenizer = BartTokenizer.from_pretrained("bart-large-cnn")
        new_df = self.update()
        pd.concat([self.df, new_df])
        self.csv_path = csv_path

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

    def update(self, split="test"):
        records = []
        to_score = set(self.finished_experiments).difference(self.rouged_experiments)
        for exp_name in tqdm_nice(to_score, desc='Rouge Update')
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
        return new_df
