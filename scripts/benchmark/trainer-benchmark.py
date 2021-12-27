#!/usr/bin/env python

# HF Trainer benchmarking tool
#
# This tool can be used to run and compare multiple dimensions of the HF Trainers args
#
# The main idea is:
# ./trainer-benchmark.py --base-cmd '<cmd args that don't change>' \
# --dims '--tf32 0 --tf32 1' '--fp16 0 --fp16 1 --bf16 1' \
# --metric-key train_samples_per_second
#
# --dims allows you to compare multiple dimensions.
#
# as the first dimension has 2 options and the second 3, this will run the trainer 6 times adding
# one of:
#
# --tf32 0 --fp16 0
# --tf32 0 --fp16 1
# --tf32 0 --bf16 1
# --tf32 1 --fp16 0
# --tf32 1 --fp16 1
# --tf32 1 --bf16 1
#
# and print the results. This is just a cartesian product - and more than 2 dimensions can be used.
#
# Here is a full example of a train:
#
# CUDA_VISIBLE_DEVICES=0 ./scripts/benchmark/trainer-benchmark.py \
# --base-cmd ' \
# examples/pytorch/translation/run_translation.py --model_name_or_path t5-small --output_dir output_dir \
# --do_train --label_smoothing 0.1 --logging_strategy no --save_strategy no --per_device_train_batch_size 8 \
# --max_source_length 512 --max_target_length 512 --num_train_epochs 1 --overwrite_output_dir \
# --source_lang en --target_lang ro --dataset_name wmt16 --dataset_config "ro-en" \
# --source_prefix "translate English to Romanian: "  --warmup_steps 50 \
# --max_train_samples 5000 --dataloader_num_workers 2 \
# ' \
# --dims '--tf32 0 --tf32 1' '--fp16 0 --fp16 1 --bf16 1' \
# --base-dim '--tf32 0 --fp16 0' \
# --metric-key train_samples_per_second --repeat-times 1
#
# and here a possible output:
#
# *** Results: train_samples_per_second
#
# |    Variations     | Result |   %   |
# | ----------------- | ------ | ----- |
# | --tf32 0 --fp16 0 |  31.95 |  100% |
# | --tf32 0 --fp16 1 |  47.88 |  149% |
# | --tf32 0 --bf16 1 |  35.04 |  109% |
# | --tf32 1 --fp16 0 |  35.47 |  111% |
# | --tf32 1 --fp16 1 |  47.82 |  149% |
# | --tf32 1 --bf16 1 |  35.11 |  109% |
#
# So you can quickly compare the different outcomes.
#
# Typically running each experiment once is enough, but if the environment is unstable you can
# re-run each multiple times, e.g., 3 using --repeat-times 3 and it will report the average results.
#
# by default it'll use the worst result as the base line to use as 100% and then compare the rest to
# it as can be seen from the table, but you can also specify which combination is the one to use as
# the baseline, e.g., to change to another entry use: --base-dim '--tf32 1 --fp16 0'
#
# --metric-key is there to tell the program which metrics to compare - the different metric keys are
# inside output_dir/all_results.json. e.g., to measure eval performance instead of train use
# --metric-key eval_samples_per_second


import argparse
import io
import itertools
import json
import re
import shlex
import subprocess
import sys
from statistics import fmean

from tqdm import tqdm


def get_base_cmd(args, output_dir):

    # unwrap multi-line input
    args.base_cmd = re.sub(r"\\", " ", args.base_cmd)
    args.base_cmd = re.sub(r"\n", " ", args.base_cmd)

    # remove --output_dir if any and set our own
    args.base_cmd = re.sub("--output_dir\s+[^\s]+", "", args.base_cmd)
    args.base_cmd += f"--output_dir {output_dir} "

    # ensure we have --overwrite_output_dir
    args.base_cmd = re.sub("--overwrite_output_dir\s+", "", args.base_cmd)
    args.base_cmd += "--overwrite_output_dir "

    return [sys.executable] + shlex.split(args.base_cmd)


def process_run(id, cmd, opt, repeat_times, output_dir, metric_key, verbose):
    results = []
    preamble = f"{id}: {opt}"
    outcome = f"{preamble}: "
    for i in tqdm(range(repeat_times), desc=preamble, leave=False):
        result = process_run_single(id, cmd, opt, output_dir, metric_key, verbose)
        if result != -1:
            results.append(result)
            outcome += "✓"
        else:
            outcome += "✘"
    outcome = f"\33[2K\r{outcome}"
    if len(results):
        mean_result = round(fmean(results), 2)
        results_str = f"{outcome} {mean_result}"
        if len(results) > 1:
            results_str += f" ({[round(x, 2) for x in results]})"
        print(results_str)
        return mean_result
    else:
        print(outcome)
        return -1


def process_run_single(id, cmd, opt, output_dir, metric_key, verbose):
    # enable to debug everything but the run itself, to do it fast and see the progress
    # from random import randint
    # from time import sleep
    # sleep(3)
    # return randint(100, 300)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if verbose:
        print("STDOUT", result.stdout)
        print("STDERR", result.stderr)

    if result.returncode != 0:
        if verbose:
            print("failed")
        return -1

    filename = f"{output_dir}/all_results.json"
    with io.open(filename, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics[metric_key]


def process_results(results, metric_key, base_dim):

    print(f"\n*** Results: {metric_key}\n")

    col_opt, col_result, col_relative = "Variations", "Result", "%"
    width_opt = max(len(k) for k in list(results.keys()) + [col_opt])
    width_metric = max(len(str(v)) for v in list(results.values()) + [col_result])
    width_percent = 5

    if base_dim is not None and base_dim in results:
        sentinel_value = results[base_dim]
    else:
        # if no match, use the minimal value as the sentinel
        sentinel_value = min(v for v in results.values() if v != -1)

    print(f"| {col_opt:^{width_opt}} | {col_result:^{width_metric}} | {col_relative:^{width_percent}} |")
    print(f"| {'-'*width_opt:{width_opt}} | {'-'*width_metric:{width_metric}} | {'-'*width_percent:{width_percent}} |")
    for key, value in results.items():
        if value != -1:
            percent = f"{int(100*value/sentinel_value)}%"
            value = f"{value:.02f}"
        else:
            percent = "✘"
            value = "✘"
        print(f"| {key:{width_opt}} | {value:>{width_metric}} | {percent:>{width_percent}} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-cmd",
        default=None,
        type=str,
        required=True,
        help="Base cmd",
    )
    parser.add_argument(
        "--dims",
        default=None,
        type=str,
        nargs="+",
        required=True,
        help="Dimension args",
    )
    parser.add_argument(
        "--base-dim",
        default=None,
        type=str,
        help="Dimension base line arg. if None the minimal value will be used to compare against",
    )
    parser.add_argument(
        "--metric-key",
        default=None,
        type=str,
        required=True,
        help="Metric key in output_dir/all_results.json, e.g., train_samples_per_second",
    )
    parser.add_argument(
        "--repeat-times",
        default=1,
        type=int,
        help="How many times to re-run each combination - an average will be reported",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Whether to show the outputs of each run or just the benchmark progress",
    )
    args = parser.parse_args()

    output_dir = "output_benchmark"
    base_cmd = get_base_cmd(args, output_dir)

    results = {}
    dims = [list(map(str.strip, re.split(r"(?=--)", x)[1:])) for x in args.dims]
    # cartesian product of dimensions and then converted back into cmd-line arg strings
    opts = list(map(" ".join, itertools.product(*dims)))

    print(f"\n*** Running {len(opts)} benchmarks:")
    print(f"Base command: {' '.join(base_cmd)}")

    for id, opt in enumerate(tqdm(opts, desc="Total completion: ", leave=False)):
        cmd = base_cmd + opt.split()
        results[opt] = process_run(id + 1, cmd, opt, args.repeat_times, output_dir, args.metric_key, args.verbose)

    process_results(results, args.metric_key, args.base_dim)


if __name__ == "__main__":
    main()
