# Copyright 2024 The HuggingFace Team and the AllenNLP authors. All rights reserved.
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
A handy script to run benchmark script(s) against a list of git commits.

Example: python utils/benchmark.py --benchmark_path src/transformers/benchmark/from_pretrained_benchmark.py --base_output_path "./bench_reports" --commits "62408e13,6c97e80b0"
"""


import argparse
import os
from contextlib import contextmanager
from pathlib import Path

from git import Repo


PATH_TO_REPO = Path(__file__).parent.parent.resolve()


@contextmanager
def checkout_commit(repo: Repo, commit_id: str):
    """
    Context manager that checks out a given commit when entered, but gets back to the reference it was at on exit.

    Args:
        repo (`git.Repo`): A git repository (for instance the Transformers repo).
        commit_id (`str`): The commit reference to checkout inside the context manager.
    """
    current_head = repo.head.commit if repo.head.is_detached else repo.head.ref

    try:
        repo.git.checkout(commit_id)
        yield

    finally:
        repo.git.checkout(current_head)


if __name__ == "__main__":

    def list_str(values):
        return values.split(",")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark_path",
        type=str,
        required=True,
        help="Path to the benchmark script to run.",
    )
    parser.add_argument(
        "--base_output_path",
        type=str,
        required=True,
        help="Base path to the output file where the run's info. will be saved.",
    )
    parser.add_argument(
        "--config_path",
        default=None,
        type=str,
        required=False,
        help="Path to a prepared run file or a previously run output file.",
    )
    parser.add_argument(
        "--commits",
        type=list_str,
        required=True,
        help="Comma-separated list of commit SHA values against which the benchmark will be run",
    )

    args = parser.parse_args()

    repo = Repo(PATH_TO_REPO)

    for commit in args.commits:
        with checkout_commit(repo, commit):
            print(f"benchmark against commit: {repo.head.commit}")

            output_path = os.path.join(args.base_output_path, f"{commit}")
            commandline_args = f"--output_path {output_path}"
            if args.config_path is not None:
                commandline_args += " --config_path {args.config_path}"

            # TODO: use `subprocess`
            os.system(f"python {args.benchmark_path} {commandline_args}")
