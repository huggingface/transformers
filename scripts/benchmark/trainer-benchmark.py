#!/usr/bin/env python

# HF Trainer benchmarking tool
#
# This tool can be used to run and compare multiple dimensions of the HF Trainers args
#
# The main idea is:
# ./trainer-benchmark.py --base-cmd '<cmd args that don't change>' \
# --dims '--tf32 0 --tf32 1' '--fp16 0 --fp16 1 --bf16 1' \
# --target-metric-key train_samples_per_second
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
# --target-metric-key train_samples_per_second --repeat-times 1
#
# and here is a possible output:
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
# re-run each multiple times, e.g., 3 using --repeat-times 3 and it will report the averaged results.
#
# By default it'll use the lowest result as the base line to use as 100% and then compare the rest to
# it as can be seen from the table above, but you can also specify which combination is the one to use as
# the baseline, e.g., to change to another entry use: --base-dim '--tf32 1 --fp16 0'
#
# --target-metric-key is there to tell the program which metrics to compare - the different metric keys are
# inside output_dir/all_results.json. e.g., to measure eval performance instead of train use
# --target-metric-key eval_samples_per_second


import argparse
import datetime
import io
import itertools
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from statistics import fmean

import pandas as pd
import torch
from tqdm import tqdm

import transformers


def get_base_cmd(args, output_dir):

    # unwrap multi-line input
    args.base_cmd = re.sub(r"[\\\n]+", " ", args.base_cmd)

    # remove --output_dir if any and set our own
    args.base_cmd = re.sub("--output_dir\s+[^\s]+", "", args.base_cmd)
    args.base_cmd += f" --output_dir {output_dir}"

    # ensure we have --overwrite_output_dir
    args.base_cmd = re.sub("--overwrite_output_dir\s+", "", args.base_cmd)
    args.base_cmd += " --overwrite_output_dir"

    return [sys.executable] + shlex.split(args.base_cmd)


def process_run_single(id, cmd, opt, output_dir, target_metric_key, metric_keys, verbose):
    # enable to debug everything but the run itself, to do it fast and see the progress
    if 0:
        import random
        from random import randint
        from time import sleep

        sleep(0)
        return dict(
            {k: randint(1, 30) for k in metric_keys}, **{target_metric_key: random.choice([-1, 10, 100, 55, 222])}
        )

    result = subprocess.run(cmd, capture_output=True, text=True)

    if verbose:
        print("STDOUT", result.stdout)
        print("STDERR", result.stderr)

    # save the streams
    prefix = opt.replace(" ", "-")
    with open(Path(output_dir) / f"{prefix}.stdout.txt", "w") as f:
        f.write(result.stdout)
    with open(Path(output_dir) / f"{prefix}.stderr.txt", "w") as f:
        f.write(result.stderr)

    if result.returncode != 0:
        if verbose:
            print("failed")
        return {target_metric_key: -1}

    with io.open(f"{output_dir}/all_results.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # filter out just the keys we want
    return {k: v for k, v in metrics.items() if k in metric_keys}


def process_run(id, cmd, opt_key, opt, target_metric_key, report_metric_keys, repeat_times, output_dir, verbose):
    results = []
    metrics = []
    preamble = f"{id}: {opt}"
    outcome = f"{preamble}: "
    metric_keys = set(report_metric_keys + [target_metric_key])
    for i in tqdm(range(repeat_times), desc=preamble, leave=False):
        single_run_metrics = process_run_single(id, cmd, opt, output_dir, target_metric_key, metric_keys, verbose)
        result = single_run_metrics[target_metric_key]
        if result != -1:
            metrics.append(single_run_metrics)
            results.append(result)
            outcome += "✓"
        else:
            outcome += "✘"
    outcome = f"\33[2K\r{outcome}"
    successful_runs = len(metrics)
    if successful_runs > 0:
        mean_metrics = {k: fmean([metrics[i][k] for i in range(successful_runs)]) for k in metrics[0].keys()}
        mean_target = round(mean_metrics[target_metric_key], 2)
        results_str = f"{outcome} {mean_target}"
        if successful_runs > 1:
            results_str += f" {tuple(round(x, 2) for x in results)}"
        print(results_str)
        mean_metrics[opt_key] = opt
        return mean_metrics
    else:
        print(outcome)
        return {opt_key: opt, target_metric_key: -1}


def get_versions():
    properties = torch.cuda.get_device_properties(torch.device("cuda"))
    return f"""
Datetime    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
transformers: {transformers.__version__}
torch       : {torch.__version__}
cuda        : {torch.version.cuda}
{torch.cuda.device_count()} GPUs      : {properties.name}, {properties.total_memory/2**30:0.2f}GB
"""


def process_results(results, target_metric_key, report_metric_keys, base_dim, table_format, output_dir):

    df = pd.DataFrame(results)
    variation_key = "variation"
    diff_key = "diff_%"

    sentinel_value = -1
    if base_dim is not None and len(df[df.variation == base_dim]):
        # this may still return -1
        sentinel_value = df.loc[df.variation == base_dim][target_metric_key]
    if sentinel_value == -1:
        # as a fallback, use the minimal value as the sentinel
        sentinel_value = df.loc[df[target_metric_key] != -1][target_metric_key].min()

    # create diff column
    if sentinel_value != -1:
        df[diff_key] = df.apply(
            lambda r: int(100 * r[target_metric_key] / sentinel_value) if r[target_metric_key] != -1 else "✘",
            axis="columns",
        )

    # deal with failed runs
    df[target_metric_key] = df.apply(
        lambda r: r[target_metric_key] if r[target_metric_key] != -1 else "✘", axis="columns"
    )

    # re-order columns
    cols = [variation_key, target_metric_key, diff_key, *report_metric_keys]
    df = df.reindex(cols, axis="columns")  # reorder cols

    # capitalize
    df = df.rename(str.capitalize, axis="columns")

    # make the cols as narrow as possible
    linebreak = "<br>" if table_format == "github" else "\n"
    df = df.rename(lambda c: c.replace("_", linebreak), axis="columns")

    print("\n*** Results:\n")
    print(df.to_markdown(index=False))
    print(f"\nNote: each run's output is also logged under {output_dir}/*.std*.txt")
    print("\nPlease include the following information with your benchmark post:")
    print(get_versions())


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
        "--target-metric-key",
        default=None,
        type=str,
        required=True,
        help="Target metric key in output_dir/all_results.json, e.g., train_samples_per_second",
    )
    parser.add_argument(
        "--report-metric-keys",
        default="",
        type=str,
        help="Report metric keys - other metric keys from output_dir/all_results.json to report, e.g., train_loss. Use a single argument e.g., 'train_loss train_samples",
    )
    parser.add_argument(
        "--repeat-times",
        default=1,
        type=int,
        help="How many times to re-run each combination - an average will be reported",
    )
    # table_format_choices
    parser.add_argument(
        "--table-format",
        default="console",
        type=str,
        choices=["github", "console"],
        help="Format the results table to render best in the destination use",
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

    # split each dimension into its --foo variations
    dims = [list(map(str.strip, re.split(r"(?=--)", x)[1:])) for x in args.dims]
    # build a cartesian product of dimensions and convert those back into cmd-line arg strings
    opts = list(map(" ".join, itertools.product(*dims)))

    # split wanted keys
    report_metric_keys = args.report_metric_keys.split()

    print(f"\n*** Running {len(opts)} benchmarks:")
    print(f"Base command: {' '.join(base_cmd)}")

    opt_key = "variation"
    results = []
    for id, opt in enumerate(tqdm(opts, desc="Total completion: ", leave=False)):
        cmd = base_cmd + opt.split()
        results.append(
            process_run(
                id + 1,
                cmd,
                opt_key,
                opt,
                args.target_metric_key,
                report_metric_keys,
                args.repeat_times,
                output_dir,
                args.verbose,
            )
        )

    process_results(results, args.target_metric_key, report_metric_keys, args.base_dim, args.table_format, output_dir)


if __name__ == "__main__":
    main()
