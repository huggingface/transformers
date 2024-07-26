#!/usr/bin/env python

# HF Trainer benchmarking tool
#
# This tool can be used to run and compare multiple dimensions of the HF Trainers args.
#
# It then prints a report once in github format with all the information that needs to be shared
# with others and second time in a console-friendly format, so it's easier to use for tuning things up.
#
# The main idea is:
#
#     ./trainer-benchmark.py --base-cmd '<cmd args that don't change>' \
#     --variations '--tf32 0|--tf32 1' '--fp16 0|--fp16 1|--bf16 1' \
#     --target-metric-key train_samples_per_second
#
# The variations can be any command line argument that you want to compare and not just dtype as in
# the example.
#
# --variations allows you to compare variations in multiple dimensions.
#
# as the first dimention has 2 options and the second 3 in our example, this will run the trainer 6
# times adding one of:
#
#    1. --tf32 0 --fp16 0
#    2. --tf32 0 --fp16 1
#    3. --tf32 0 --bf16 1
#    4. --tf32 1 --fp16 0
#    5. --tf32 1 --fp16 1
#    6. --tf32 1 --bf16 1
#
# and print the results. This is just a cartesian product - and more than 2 dimensions can be used.
#
# If you want to rely on defaults, this:
#    --variations '--tf32 0|--tf32 1' '--fp16 0|--fp16 1|--bf16 1'
# is identical to this:
#    --variations '--tf32 0|--tf32 1' '|--fp16|--bf16'
#
# the leading empty variation in the 2nd dimension is a valid variation.
#
# So here we get the following 6 variations:
#
#    1. --tf32 0
#    2. --tf32 0 --fp16
#    3. --tf32 0 --bf16
#    4. --tf32 1
#    5. --tf32 1 --fp16
#    6. --tf32 1 --bf16
#
# In this particular case we don't know what the default tf32 setting is as it's normally
# pytorch-version dependent). That's why it's best to do an explicit setting of each variation:
#    `--tf32 0|--tf32 1`
#
# Here is a full example of a train:
#
# CUDA_VISIBLE_DEVICES=0 python ./scripts/benchmark/trainer-benchmark.py \
# --base-cmd \
# ' examples/pytorch/translation/run_translation.py --model_name_or_path google-t5/t5-small \
# --output_dir output_dir --do_train --label_smoothing 0.1 --logging_strategy no \
# --save_strategy no --per_device_train_batch_size 32 --max_source_length 512 \
# --max_target_length 512 --num_train_epochs 1 --overwrite_output_dir \
# --source_lang en --target_lang ro --dataset_name wmt16 --dataset_config "ro-en" \
# --source_prefix "translate English to Romanian: " --warmup_steps 50 \
# --max_train_samples 20000 --dataloader_num_workers 2 ' \
# --target-metric-key train_samples_per_second --repeat-times 1 --variations \
# '|--fp16|--bf16' '--tf32 0|--tf32 1' --report-metric-keys train_loss \
# --repeat-times 1 --base-variation '--tf32 0'
#
# and here is a possible output:
#
#
# | Variation       |     Train |   Diff |   Train |
# |                 |   samples |      % |    loss |
# |                 |       per |        |         |
# |                 |    second |        |         |
# |:----------------|----------:|-------:|--------:|
# | --tf32 0        |    285.11 |      0 |    2.51 |
# | --tf32 1        |    342.09 |     20 |    2.51 |
# | --fp16 --tf32 0 |    423.49 |     49 |    2.51 |
# | --fp16 --tf32 1 |    423.13 |     48 |    2.51 |
# | --bf16 --tf32 0 |    416.80 |     46 |    2.52 |
# | --bf16 --tf32 1 |    415.87 |     46 |    2.52 |
#
#
# So you can quickly compare the different outcomes.
#
# Typically running each experiment once is enough, but if the environment is unstable you can
# re-run each multiple times, e.g., 3 using --repeat-times 3 and it will report the averaged results.
#
# By default it'll use the lowest result as the base line to use as 100% and then compare the rest to
# it as can be seen from the table above, but you can also specify which combination is the one to use as
# the baseline, e.g., to change to another entry use: --base-variation '--tf32 1 --fp16 0'
#
# --target-metric-key is there to tell the program which metrics to compare - the different metric keys are
# inside output_dir/all_results.json. e.g., to measure eval performance instead of train use:
#    --target-metric-key eval_samples_per_second
# but of course you will need to adjust the --base-cmd value in the example to perform evaluation as
# well (as currently it doesn't)
#

import argparse
import datetime
import io
import itertools
import json
import math
import os
import platform
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


nan = float("nan")


class Tee:
    """
    A helper class to tee print's output into a file.
    Usage:
    sys.stdout = Tee(filename)
    """

    def __init__(self, filename):
        self.stdout = sys.stdout
        self.file = open(filename, "a")

    def __getattr__(self, attr):
        return getattr(self.stdout, attr)

    def write(self, msg):
        self.stdout.write(msg)
        # strip tqdm codes
        self.file.write(re.sub(r"^.*\r", "", msg, 0, re.M))


def get_original_command(max_width=80, full_python_path=False):
    """
    Return the original command line string that can be replayed nicely and wrapped for 80 char width.

    Args:
        max_width (`int`, *optional*, defaults to 80):
            The width to wrap for.
        full_python_path (`bool`, `optional`, defaults to `False`):
             Whether to replicate the full path or just the last segment (i.e. `python`).
    """

    cmd = []

    # deal with critical env vars
    env_keys = ["CUDA_VISIBLE_DEVICES"]
    for key in env_keys:
        val = os.environ.get(key, None)
        if val is not None:
            cmd.append(f"{key}={val}")

    # python executable (not always needed if the script is executable)
    python = sys.executable if full_python_path else sys.executable.split("/")[-1]
    cmd.append(python)

    # now the normal args
    cmd += list(map(shlex.quote, sys.argv))

    # split up into up to MAX_WIDTH lines with shell multi-line escapes
    lines = []
    current_line = ""
    while len(cmd) > 0:
        current_line += f"{cmd.pop(0)} "
        if len(cmd) == 0 or len(current_line) + len(cmd[0]) + 1 > max_width - 1:
            lines.append(current_line)
            current_line = ""
    return "\\\n".join(lines)


def get_base_command(args, output_dir):

    # unwrap multi-line input
    args.base_cmd = re.sub(r"[\\\n]+", " ", args.base_cmd)

    # remove --output_dir if any and set our own
    args.base_cmd = re.sub("--output_dir\s+[^\s]+", "", args.base_cmd)
    args.base_cmd += f" --output_dir {output_dir}"

    # ensure we have --overwrite_output_dir
    args.base_cmd = re.sub("--overwrite_output_dir\s+", "", args.base_cmd)
    args.base_cmd += " --overwrite_output_dir"

    return [sys.executable] + shlex.split(args.base_cmd)


def process_run_single(id, cmd, variation, output_dir, target_metric_key, metric_keys, verbose):

    # Enable to debug everything but the run itself, to do it fast and see the progress.
    # This is useful for debugging the output formatting quickly - we can remove it later once
    # everybody is happy with the output
    if 0:
        import random
        from time import sleep

        sleep(0)
        return dict(
            {k: random.uniform(0, 100) for k in metric_keys},
            **{target_metric_key: random.choice([nan, 10.31, 100.2, 55.6666, 222.22222222])},
        )

    result = subprocess.run(cmd, capture_output=True, text=True)

    if verbose:
        print("STDOUT", result.stdout)
        print("STDERR", result.stderr)

    # save the streams
    prefix = variation.replace(" ", "-")
    with open(Path(output_dir) / f"log.{prefix}.stdout.txt", "w") as f:
        f.write(result.stdout)
    with open(Path(output_dir) / f"log.{prefix}.stderr.txt", "w") as f:
        f.write(result.stderr)

    if result.returncode != 0:
        if verbose:
            print("failed")
        return {target_metric_key: nan}

    with io.open(f"{output_dir}/all_results.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # filter out just the keys we want
    return {k: v for k, v in metrics.items() if k in metric_keys}


def process_run(
    id,
    cmd,
    variation_key,
    variation,
    longest_variation_len,
    target_metric_key,
    report_metric_keys,
    repeat_times,
    output_dir,
    verbose,
):
    results = []
    metrics = []
    preamble = f"{id}: {variation:<{longest_variation_len}}"
    outcome = f"{preamble}: "
    metric_keys = set(report_metric_keys + [target_metric_key])
    for i in tqdm(range(repeat_times), desc=preamble, leave=False):
        single_run_metrics = process_run_single(
            id, cmd, variation, output_dir, target_metric_key, metric_keys, verbose
        )
        result = single_run_metrics[target_metric_key]
        if not math.isnan(result):
            metrics.append(single_run_metrics)
            results.append(result)
            outcome += "✓"
        else:
            outcome += "✘"
    outcome = f"\33[2K\r{outcome}"
    if len(metrics) > 0:
        mean_metrics = {k: fmean([x[k] for x in metrics]) for k in metrics[0].keys()}
        mean_target = round(mean_metrics[target_metric_key], 2)
        results_str = f"{outcome} {mean_target}"
        if len(metrics) > 1:
            results_str += f" {tuple(round(x, 2) for x in results)}"
        print(results_str)
        mean_metrics[variation_key] = variation
        return mean_metrics
    else:
        print(outcome)
        return {variation_key: variation, target_metric_key: nan}


def get_versions():
    properties = torch.cuda.get_device_properties(torch.device("cuda"))
    return f"""
Datetime    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Software:
transformers: {transformers.__version__}
torch       : {torch.__version__}
cuda        : {torch.version.cuda}
python      : {platform.python_version()}

Hardware:
{torch.cuda.device_count()} GPUs      : {properties.name}, {properties.total_memory/2**30:0.2f}GB
"""


def process_results(results, target_metric_key, report_metric_keys, base_variation, output_dir):

    df = pd.DataFrame(results)
    variation_key = "variation"
    diff_key = "diff_%"

    sentinel_value = nan
    if base_variation is not None and len(df[df[variation_key] == base_variation]):
        # this may still return nan
        sentinel_value = df.loc[df[variation_key] == base_variation][target_metric_key].item()
    if math.isnan(sentinel_value):
        # as a fallback, use the minimal value as the sentinel
        sentinel_value = df.loc[df[target_metric_key] != nan][target_metric_key].min()

    # create diff column if possible
    if not math.isnan(sentinel_value):
        df[diff_key] = df.apply(
            lambda r: round(100 * (r[target_metric_key] - sentinel_value) / sentinel_value)
            if not math.isnan(r[target_metric_key])
            else 0,
            axis="columns",
        )

    # re-order columns
    cols = [variation_key, target_metric_key, diff_key, *report_metric_keys]
    df = df.reindex(cols, axis="columns")  # reorder cols

    # capitalize
    df = df.rename(str.capitalize, axis="columns")

    # make the cols as narrow as possible
    df_github = df.rename(lambda c: c.replace("_", "<br>"), axis="columns")
    df_console = df.rename(lambda c: c.replace("_", "\n"), axis="columns")

    report = ["", "Copy between the cut-here-lines and paste as is to github or a forum"]
    report += ["----------8<-----------------8<--------"]
    report += ["*** Results:", df_github.to_markdown(index=False, floatfmt=".2f")]
    report += ["```"]
    report += ["*** Setup:", get_versions()]
    report += ["*** The benchmark command line was:", get_original_command()]
    report += ["```"]
    report += ["----------8<-----------------8<--------"]
    report += ["*** Results (console):", df_console.to_markdown(index=False, floatfmt=".2f")]

    print("\n\n".join(report))


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
        "--variations",
        default=None,
        type=str,
        nargs="+",
        required=True,
        help="Multi-dimensional variations, example: '|--fp16|--bf16' '|--tf32'",
    )
    parser.add_argument(
        "--base-variation",
        default=None,
        type=str,
        help="Baseline variation to compare to. if None the minimal target value will be used to compare against",
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
        help="How many times to re-run each variation - an average will be reported",
    )
    parser.add_argument(
        "--output_dir",
        default="output_benchmark",
        type=str,
        help="The output directory where all the benchmark reports will go to and additionally this directory will be used to override --output_dir in the script that is being benchmarked",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Whether to show the outputs of each run or just the benchmark progress",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    Path(output_dir).mkdir(exist_ok=True)
    base_cmd = get_base_command(args, output_dir)

    # split each dimension into its --foo variations
    dims = [list(map(str.strip, re.split(r"\|", x))) for x in args.variations]
    # build a cartesian product of dimensions and convert those back into cmd-line arg strings,
    # while stripping white space for inputs that were empty
    variations = list(map(str.strip, map(" ".join, itertools.product(*dims))))
    longest_variation_len = max(len(x) for x in variations)

    # split wanted keys
    report_metric_keys = args.report_metric_keys.split()

    # capture prints into a log file for convenience
    report_fn = f"benchmark-report-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt"
    print(f"\nNote: each run's output is also logged under {output_dir}/log.*.std*.txt")
    print(f"and this script's output is also piped into {report_fn}")

    sys.stdout = Tee(report_fn)

    print(f"\n*** Running {len(variations)} benchmarks:")
    print(f"Base command: {' '.join(base_cmd)}")

    variation_key = "variation"
    results = []
    for id, variation in enumerate(tqdm(variations, desc="Total completion: ", leave=False)):
        cmd = base_cmd + variation.split()
        results.append(
            process_run(
                id + 1,
                cmd,
                variation_key,
                variation,
                longest_variation_len,
                args.target_metric_key,
                report_metric_keys,
                args.repeat_times,
                output_dir,
                args.verbose,
            )
        )

    process_results(results, args.target_metric_key, report_metric_keys, args.base_variation, output_dir)


if __name__ == "__main__":
    main()
