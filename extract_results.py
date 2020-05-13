""" Go through a directory containing output directories from runs and pull out results in eval_results.txt """

import os
import re
import pdb
from collections import defaultdict

import numpy as np


TASK = "multirc"
TASK2METRIC = {
               "copa": "acc",
               "multirc": "f1", #"em_and_f1",
               "wsc": "f1" #"acc_and_f1"
              }
METRIC = TASK2METRIC[TASK]

def get_result(data_file):
    lines = open(data_file, "r").readlines()
    p = re.compile(f"^{METRIC} =")
    metric_line = list(filter(lambda x: p.match(x), lines))
    assert len(metric_line) == 1, "Multiple lines found"
    return float(metric_line[0].strip().split("=")[1])

DATA_DIR = f"/scratch/aw3272/ckpts/transformers/superglue/roberta-large/{TASK}"
run2score = {}
score2run = defaultdict(list)
for run_dir in os.listdir(DATA_DIR):
    result_file = os.path.join(DATA_DIR, run_dir, "eval_results.txt")
    if os.path.exists(result_file):
        score = get_result(result_file)
        run2score[run_dir] = score
        score2run[score].append(run_dir)

# max score
max_score = max(run2score.values())
max_run = score2run[max_score]
print(f"Max {METRIC}: {max_score} by {max_run}")

# results by setting
setting2seed = defaultdict(dict)
for run, score in run2score.items():
    settings = run.split("-")
    setting2seed[",".join(settings[:-1])][settings[-1]] = score

settings = sorted(list(setting2seed.keys()))
for setting in settings:
    seed2score = setting2seed[setting]
    seeds = sorted(list(seed2score.keys()))
    scores = np.array([seed2score[seed] for seed in seeds])
    print(f"Setting {setting} ({len(seed2score)} runs)")
    print(f"\tmax {METRIC} {np.max(scores):.3f}")
    print(f"\tmean {METRIC} {np.mean(scores):.3f}")
    print(f"\tstd {METRIC} {np.std(scores):.3f}")
