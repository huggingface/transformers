# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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

from git import Repo

repo = Repo(".")
branching_commits = repo.merge_base(repo.refs.master, repo.head)

for commit in branching_commits:
    print(f"Branching commit: {commit}")

print(f"Master is at {repo.refs.master.commit}")
print(f"Current head is at {repo.head.commit}")

print("### DIFF ###")
for commit in branching_commits:
    for diff_obj in commit.diff(repo.head.commit):
        print(diff_obj.change_type, diff_obj.a_path, diff_obj.b_path, diff_obj)