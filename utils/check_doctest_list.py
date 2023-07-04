# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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

import os


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_doctest_list.py
REPO_PATH = "."


if __name__ == "__main__":
    doctest_file_path = os.path.join(REPO_PATH, "utils/documentation_tests.txt")
    non_existent_paths = []
    all_paths = []
    with open(doctest_file_path) as fp:
        for line in fp:
            line = line.strip()
            path = os.path.join(REPO_PATH, line)
            if not (os.path.isfile(path) or os.path.isdir(path)):
                non_existent_paths.append(line)
            all_paths.append(path)
    if len(non_existent_paths) > 0:
        non_existent_paths = "\n".join(non_existent_paths)
        raise ValueError(f"`utils/documentation_tests.txt` contains non-existent paths:\n{non_existent_paths}")
    if all_paths != sorted(all_paths):
        raise ValueError("Files in `utils/documentation_tests.txt` are not in alphabetical order.")
