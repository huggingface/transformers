#!/usr/bin/env python

import io
import json
import subprocess


pairs = [
    ["en", "ru"],
    ["ru", "en"],
    ["en", "de"],
    ["de", "en"],
]

n_objs = 8


def get_all_data(pairs, n_objs):
    text = {}
    for src, tgt in pairs:
        pair = f"{src}-{tgt}"
        cmd = f"sacrebleu -t wmt19 -l {pair} --echo src".split()
        src_lines = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode("utf-8").splitlines()
        cmd = f"sacrebleu -t wmt19 -l {pair} --echo ref".split()
        tgt_lines = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode("utf-8").splitlines()
        text[pair] = {"src": src_lines[:n_objs], "tgt": tgt_lines[:n_objs]}
    return text


text = get_all_data(pairs, n_objs)
filename = "./fsmt_val_data.json"
with io.open(filename, "w", encoding="utf-8") as f:
    bleu_data = json.dump(text, f, indent=2, ensure_ascii=False)
