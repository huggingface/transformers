# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import re


PATH_TO_EXAMPLES = "examples/"
_re_min_version = re.compile(r'^check_min_version\("([^"]+)"\)\s*$', re.MULTILINE)


def bump_version_in_file(file, version):
    with open(file, "r") as f:
        code = f.read()
    code = _re_min_version.sub(f'check_min_version("{version}")\n', code)
    with open(file, "w") as f:
        f.write(code)


def bump_version_in_examples(version):
    for folder, directories, files in os.walk(PATH_TO_EXAMPLES):
        # Removing some of the folders with non-actively maintained examples from the walk
        if "research_projects" in directories:
            directories.remove("research_projects")
        if "legacy" in directories:
            directories.remove("legacy")
        for file in files:
            if file.endswith(".py"):
                print(os.path.join(folder, file))
                bump_version_in_file(os.path.join(folder, file), version)