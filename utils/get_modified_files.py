# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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

# this script reports modified .py files under the desired list of top-level sub-dirs passed as a list of arguments, e.g.:
#   python ./utils/get_modified_files.py utils src tests examples
#
# it uses git to find the forking point and which files were modified - i.e. files not under git won't be considered
# since the output of this script is fed into Makefile commands it doesn't print a newline after the results

import re
import subprocess
import sys


fork_point_sha = subprocess.check_output("git merge-base main HEAD".split()).decode("utf-8")
modified_files = subprocess.check_output(f"git diff --name-only {fork_point_sha}".split()).decode("utf-8").split()

joined_dirs = "|".join(sys.argv[1:])
regex = re.compile(rf"^({joined_dirs}).*?\.py$")

relevant_modified_files = [x for x in modified_files if regex.match(x)]
print(" ".join(relevant_modified_files), end="")
