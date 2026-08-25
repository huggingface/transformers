# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is used to get the list of folders under `tests/models` and split the list into `NUM_SLICES` splits.
The main use case is a GitHub Actions workflow file calling this script to get the (nested) list of folders allowing it
to split the list of jobs to run into multiple slices each containing a smaller number of jobs. This way, we can bypass
the maximum of 256 jobs in a matrix.

The other directories under `tests` are appended to the list too, except for two groups. The ones holding no test file
at all (see `NO_TEST_FILES`) are dropped because a job for them would collect nothing. The ones a dedicated job runs in
full (see `COVERED_BY_DEDICATED_JOB`) are dropped because keeping them here would run their tests twice per CI run, on
every machine type.

See the `setup` and `run_models_gpu` jobs defined in the reusable workflow file
`.github/workflows/daily-ci_reusable.yml` in the `huggingface/transformers-ci` repository (called from
`.github/workflows/self-scheduled-caller.yml` here) for more details.

Usage:

This script is required to be run under `tests` folder of `transformers` root directory.

Assume we are under `transformers` root directory:
```bash
cd tests
python ../utils/split_model_tests.py --num_splits 64
```
"""

import argparse
import ast
import os


# Directories under `tests` (other than `models`) that a dedicated daily CI job already covers in full, skipped to avoid
# running the same tests twice per CI run, on every machine type. Opt-out of the skip with --no-skip-dedicated-jobs.
COVERED_BY_DEDICATED_JOB = ["pipelines", "quantization", "trainer"]

# Directories under `tests` holding no test file at all, so a job for them collects nothing.
NO_TEST_FILES = ["fixtures"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subdirs",
        type=str,
        default="",
        help="the list of pre-computed model names (directory names under `tests/models`) or directory names under `tests` (except `models`).",
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=1,
        help="the number of splits into which the (flat) list of folders will be split.",
    )
    parser.add_argument(
        "--no-skip-dedicated-jobs",
        action="store_true",
        help="do not skip the directories a dedicated job already covers.",
    )
    args = parser.parse_args()

    tests = os.getcwd()
    model_tests = os.listdir(os.path.join(tests, "models"))
    d1 = sorted(filter(os.path.isdir, os.listdir(tests)))
    d2 = sorted(filter(os.path.isdir, [f"models/{x}" for x in model_tests]))
    d1.remove("models")
    # Only for the auto-discovered directories: an explicit `--subdirs` request below is honored as-is.
    skipped = set(NO_TEST_FILES) | (set() if args.no_skip_dedicated_jobs else set(COVERED_BY_DEDICATED_JOB))
    d1 = [x for x in d1 if x not in skipped]
    d = d2 + d1

    if args.subdirs != "":
        model_tests = ast.literal_eval(args.subdirs)
        # We handle both cases with and without prefix because `push-important-models.yml` returns the list without
        # the prefix (i.e. `models`) but `utils/pr_slow_ci_models.py` (called by `self-comment-ci.yml`) returns the
        # list with the prefix (`models`) and some directory names under `tests`.
        d = []
        for x in model_tests:
            if os.path.isdir(x):
                d.append(x)
            if os.path.isdir(f"models/{x}"):
                d.append(f"models/{x}")
        d = sorted(d)

    num_jobs = len(d)
    num_jobs_per_splits = num_jobs // args.num_splits

    model_splits = []
    end = 0
    for idx in range(args.num_splits):
        start = end
        end = start + num_jobs_per_splits + (1 if idx < num_jobs % args.num_splits else 0)
        # Only add the slice if it is not an empty list
        if len(d[start:end]) > 0:
            model_splits.append(d[start:end])

    print(model_splits)
