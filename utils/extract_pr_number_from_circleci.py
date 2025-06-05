# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
"""Used by `.github/workflows/trigger_circleci.yml` to get the pull request number in CircleCI job runs."""

import os


if __name__ == "__main__":
    pr_number = ""

    pr = os.environ.get("CIRCLE_PULL_REQUEST", "")
    if len(pr) > 0:
        pr_number = pr.split("/")[-1]
    if pr_number == "":
        pr = os.environ.get("CIRCLE_BRANCH", "")
        if pr.startswith("pull/"):
            pr_number = "".join(pr.split("/")[1:2])

    print(pr_number)
