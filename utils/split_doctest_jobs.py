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
This script is used to get the files against which we will run doc testing.
This uses `tests_fetcher.get_all_doctest_files` then groups the test files by their directory paths.

The files in `docs/source/en/model_doc` or `docs/source/en/tasks` are **NOT** grouped together with other files in the
same directory: the objective is to run doctest against them in independent GitHub Actions jobs.

Assume we are under `transformers` root directory:
To get a map (dictionary) between directory (or file) paths and the corresponding files
```bash
python utils/split_doctest_jobs.py
```
or to get a list of lists of directory (or file) paths
```bash
python utils/split_doctest_jobs.py --only_return_keys --num_splits 4
```
(this is used to allow GitHub Actions to generate more than 256 jobs using matrix)
"""

import argparse
from collections import defaultdict
from pathlib import Path

from tests_fetcher import get_all_doctest_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only_return_keys",
        action="store_true",
        help="if to only return the keys (which is a list of list of files' directory or file paths).",
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=1,
        help="the number of splits into which the (flat) list of direcotry/file paths will be split. This has effect only if `only_return_keys` is `True`.",
    )
    args = parser.parse_args()

    all_doctest_files = get_all_doctest_files()

    raw_test_collection_map = defaultdict(list)

    for file in all_doctest_files:
        file_dir = "/".join(Path(file).parents[0].parts)
        raw_test_collection_map[file_dir].append(file)

    refined_test_collection_map = {}
    for file_dir in raw_test_collection_map.keys():
        if file_dir in ["docs/source/en/model_doc", "docs/source/en/tasks"]:
            for file in raw_test_collection_map[file_dir]:
                refined_test_collection_map[file] = file
        else:
            refined_test_collection_map[file_dir] = " ".join(sorted(raw_test_collection_map[file_dir]))

    sorted_file_dirs = sorted(refined_test_collection_map.keys())

    test_collection_map = {}
    for file_dir in sorted_file_dirs:
        test_collection_map[file_dir] = refined_test_collection_map[file_dir]

    num_jobs = len(test_collection_map)
    num_jobs_per_splits = num_jobs // args.num_splits

    file_directory_splits = []
    end = 0
    for idx in range(args.num_splits):
        start = end
        end = start + num_jobs_per_splits + (1 if idx < num_jobs % args.num_splits else 0)
        file_directory_splits.append(sorted_file_dirs[start:end])

    if args.only_return_keys:
        print(file_directory_splits)
    else:
        print(dict(test_collection_map))
