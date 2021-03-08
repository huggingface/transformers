# Copyright 2021 The HuggingFace Team, the AllenNLP library authors. All rights reserved.
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
Script to close stale issue. Taken in part from the AllenNLP repository.
https://github.com/allenai/allennlp.
"""
from datetime import datetime as dt
import os

from github import Github


LABELS_TO_EXEMPT = [
    "good first issue",
    "good second issue",
    "feature request",
    "new model",
]


def main():
    g = Github(os.environ["GITHUB_TOKEN"])
    repo = g.get_repo("huggingface/transformers")
    open_issues = repo.get_issues(state="open")

    for issue in open_issues:
        if (
            not issue.assignees
            and (dt.utcnow() - issue.updated_at).days > 21
            and (dt.utcnow() - issue.created_at).days >= 30
            and not any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels())
        ):
            print("Closing", issue)
            # issue.create_comment(
            #     "This issue has been automatically marked as stale and been closed because it has not had "
            #     "recent activity. Thank you for your contributions.\n\nIf you think this still needs to be addressed"
            #     " please comment on this thread."
            # )
            # issue.add_to_labels("wontfix")
            # issue.edit(state="closed")
        elif (
            len(issue.assignees) > 0
            and (dt.utcnow() - issue.updated_at).days > 21
            and (dt.utcnow() - issue.created_at).days >= 30
        ):
            for assignee in issue.assignees:
                print(f"Issue {issue.number}. Pinging {assignee.name} with message")
                print(f"Hey @{assignee.login}, could you take a second look at this issue?")

                # issue.create_comment(
                #    f"Hey @{assignee.login}, could you take a second look at this issue?"
                # )


if __name__ == "__main__":
    main()
