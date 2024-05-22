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

See the `setup` and `run_models_gpu` jobs defined in the workflow file `.github/workflows/self-scheduled.yml` for more
details.

Usage:

This script is required to be run under `tests` folder of `transformers` root directory.

Assume we are under `transformers` root directory:
```bash
cd tests
python ../utils/split_model_tests.py --num_splits 64
```
"""

import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_splits",
        type=int,
        default=1,
        help="the number of splits into which the (flat) list of folders will be split.",
    )
    args = parser.parse_args()

    tests = os.getcwd()
    model_tests = os.listdir(os.path.join(tests, "models"))
    d1 = sorted(filter(os.path.isdir, os.listdir(tests)))
    d2 = sorted(filter(os.path.isdir, [f"models/{x}" for x in model_tests]))
    d1.remove("models")
    d = d2 + d1

    num_jobs = len(d)
    num_jobs_per_splits = num_jobs // args.num_splits

    model_splits = []
    end = 0
    for idx in range(args.num_splits):
        start = end
        end = start + num_jobs_per_splits + (1 if idx < num_jobs % args.num_splits else 0)
        model_splits.append(d[start:end])
    model_splits = [["models/vit"]]
    print(model_splits)
