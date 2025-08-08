# Copyright 2025 The HuggingFace Team. All rights reserved.
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
This script is used to get a map containing the information of runners to use in GitHub Actions workflow files.
This is meant to be a temporary file that helps us to switch progressively from T4 to A10 runners.

The data is stored in a Hub repository [hf-internal-testing/transformers_daily_ci](https://huggingface.co/datasets/hf-internal-testing/transformers_daily_ci/blob/main/runner_map.json).
Currently, in that file, we specify the models for which we want to run the tests with T4 runners to avoid many test failures showing on the CI reports.
We will work on the tests toward to use A10 for all CI jobs.
"""

import os

import requests


if __name__ == "__main__":
    # T4
    t4_runners = {
        "single-gpu": "aws-g4dn-4xlarge-cache",
        "multi-gpu": "aws-g4dn-12xlarge-cache",
    }

    # A10
    a10_runners = {
        "single-gpu": "aws-g5-4xlarge-cache",
        "multi-gpu": "aws-g5-12xlarge-cache",
    }

    tests = os.getcwd()
    model_tests = os.listdir(os.path.join(tests, "models"))
    d1 = sorted(filter(os.path.isdir, os.listdir(tests)))
    d2 = sorted(filter(os.path.isdir, [f"models/{x}" for x in model_tests]))
    d1.remove("models")
    d = d2 + d1

    response = requests.get(
        "https://huggingface.co/datasets/hf-internal-testing/transformers_daily_ci/resolve/main/runner_map.json"
    )
    # The models that we want to run with T4 runners
    jobs_using_t4 = response.json()

    runner_map = {}
    for key in d:
        modified_key = key
        if modified_key.startswith("models/"):
            modified_key = key[len("models/") :]
        if modified_key in jobs_using_t4:
            runner_map[key] = t4_runners
        else:
            runner_map[key] = a10_runners

    print(runner_map)
