# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import collections
import json
import math
import os
import re
import time
from fnmatch import fnmatch
from typing import Dict, List

import requests
from slack_sdk import WebClient


client = WebClient(token=os.environ["CI_SLACK_BOT_TOKEN"])


def handle_test_results(test_results):
    expressions = test_results.split(" ")

    failed = 0
    success = 0

    # When the output is short enough, the output is surrounded by = signs: "== OUTPUT =="
    # When it is too long, those signs are not present.
    time_spent = expressions[-2] if "=" in expressions[-1] else expressions[-1]

    for i, expression in enumerate(expressions):
        if "failed" in expression:
            failed += int(expressions[i - 1])
        if "passed" in expression:
            success += int(expressions[i - 1])

    return failed, success, time_spent


def extract_first_line_failure(failures_short_lines):
    failures = {}
    file = None
    in_error = False
    for line in failures_short_lines.split("\n"):
        if re.search(r"_ \[doctest\]", line):
            in_error = True
            file = line.split(" ")[2]
        elif in_error and not line.split(" ")[0].isdigit():
            failures[file] = line
            in_error = False

    return failures


class Message:
    def __init__(self, title: str, doc_test_results: Dict):
        self.title = title

        self._time_spent = doc_test_results["time_spent"].split(",")[0]
        self.n_success = doc_test_results["success"]
        self.n_failures = doc_test_results["failures"]
        self.n_tests = self.n_success + self.n_failures

        # Failures and success of the modeling tests
        self.doc_test_results = doc_test_results

    @property
    def time(self) -> str:
        time_spent = [self._time_spent]
        total_secs = 0

        for time in time_spent:
            time_parts = time.split(":")

            # Time can be formatted as xx:xx:xx, as .xx, or as x.xx if the time spent was less than a minute.
            if len(time_parts) == 1:
                time_parts = [0, 0, time_parts[0]]

            hours, minutes, seconds = int(time_parts[0]), int(time_parts[1]), float(time_parts[2])
            total_secs += hours * 3600 + minutes * 60 + seconds

        hours, minutes, seconds = total_secs // 3600, (total_secs % 3600) // 60, total_secs % 60
        return f"{int(hours)}h{int(minutes)}m{int(seconds)}s"

    @property
    def header(self) -> Dict:
        return {"type": "header", "text": {"type": "plain_text", "text": self.title}}

    @property
    def no_failures(self) -> Dict:
        return {
            "type": "section",
            "text": {
                "type": "plain_text",
                "text": f"🌞 There were no failures: all {self.n_tests} tests passed. The suite ran in {self.time}.",
                "emoji": True,
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Check Action results", "emoji": True},
                "url": f"https://github.com/huggingface/transformers/actions/runs/{os.environ['GITHUB_RUN_ID']}",
            },
        }

    @property
    def failures(self) -> Dict:
        return {
            "type": "section",
            "text": {
                "type": "plain_text",
                "text": (
                    f"There were {self.n_failures} failures, out of {self.n_tests} tests.\nThe suite ran in"
                    f" {self.time}."
                ),
                "emoji": True,
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Check Action results", "emoji": True},
                "url": f"https://github.com/huggingface/transformers/actions/runs/{os.environ['GITHUB_RUN_ID']}",
            },
        }

    @property
    def category_failures(self) -> List[Dict]:
        failure_blocks = []

        MAX_ERROR_TEXT = 3000 - len("The following examples had failures:\n\n\n\n") - len("[Truncated]\n")
        line_length = 40
        category_failures = {k: v["failed"] for k, v in doc_test_results.items() if isinstance(v, dict)}

        def single_category_failures(category, failures):
            text = ""
            if len(failures) == 0:
                return ""
            text += f"*{category} failures*:".ljust(line_length // 2).rjust(line_length // 2) + "\n"

            for idx, failure in enumerate(failures):
                new_text = text + f"`{failure}`\n"
                if len(new_text) > MAX_ERROR_TEXT:
                    text = text + "[Truncated]\n"
                    break
                text = new_text

            return text

        for category, failures in category_failures.items():
            report = single_category_failures(category, failures)
            if len(report) == 0:
                continue
            block = {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"The following examples had failures:\n\n\n{report}\n",
                },
            }
            failure_blocks.append(block)

        return failure_blocks

    @property
    def payload(self) -> str:
        blocks = [self.header]

        if self.n_failures > 0:
            blocks.append(self.failures)

        if self.n_failures > 0:
            blocks.extend(self.category_failures)

        if self.n_failures == 0:
            blocks.append(self.no_failures)

        return json.dumps(blocks)

    @staticmethod
    def error_out():
        payload = [
            {
                "type": "section",
                "text": {
                    "type": "plain_text",
                    "text": "There was an issue running the tests.",
                },
                "accessory": {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Check Action results", "emoji": True},
                    "url": f"https://github.com/huggingface/transformers/actions/runs/{os.environ['GITHUB_RUN_ID']}",
                },
            }
        ]

        print("Sending the following payload")
        print(json.dumps({"blocks": json.loads(payload)}))

        client.chat_postMessage(
            channel=os.environ["CI_SLACK_CHANNEL_ID_DAILY"],
            text="There was an issue running the tests.",
            blocks=payload,
        )

    def post(self):
        print("Sending the following payload")
        print(json.dumps({"blocks": json.loads(self.payload)}))

        text = f"{self.n_failures} failures out of {self.n_tests} tests," if self.n_failures else "All tests passed."

        self.thread_ts = client.chat_postMessage(
            channel=os.environ["CI_SLACK_CHANNEL_ID_DAILY"],
            blocks=self.payload,
            text=text,
        )

    def get_reply_blocks(self, job_name, job_link, failures, text):
        # `text` must be less than 3001 characters in Slack SDK
        # keep some room for adding "[Truncated]" when necessary
        MAX_ERROR_TEXT = 3000 - len("[Truncated]")

        failure_text = ""
        for key, value in failures.items():
            new_text = failure_text + f"*{key}*\n_{value}_\n\n"
            if len(new_text) > MAX_ERROR_TEXT:
                # `failure_text` here has length <= 3000
                failure_text = failure_text + "[Truncated]"
                break
            # `failure_text` here has length <= MAX_ERROR_TEXT
            failure_text = new_text

        title = job_name
        content = {"type": "section", "text": {"type": "mrkdwn", "text": text}}

        if job_link is not None:
            content["accessory"] = {
                "type": "button",
                "text": {"type": "plain_text", "text": "GitHub Action job", "emoji": True},
                "url": job_link,
            }

        return [
            {"type": "header", "text": {"type": "plain_text", "text": title.upper(), "emoji": True}},
            content,
            {"type": "section", "text": {"type": "mrkdwn", "text": failure_text}},
        ]

    def post_reply(self):
        if self.thread_ts is None:
            raise ValueError("Can only post reply if a post has been made.")

        job_link = self.doc_test_results.pop("job_link")
        self.doc_test_results.pop("failures")
        self.doc_test_results.pop("success")
        self.doc_test_results.pop("time_spent")

        sorted_dict = sorted(self.doc_test_results.items(), key=lambda t: t[0])
        for job, job_result in sorted_dict:
            if len(job_result["failures"]):
                text = f"*Num failures* :{len(job_result['failed'])} \n"
                failures = job_result["failures"]
                blocks = self.get_reply_blocks(job, job_link, failures, text=text)

                print("Sending the following reply")
                print(json.dumps({"blocks": blocks}))

                client.chat_postMessage(
                    channel=os.environ["CI_SLACK_CHANNEL_ID_DAILY"],
                    text=f"Results for {job}",
                    blocks=blocks,
                    thread_ts=self.thread_ts["ts"],
                )

                time.sleep(1)


def get_job_links():
    run_id = os.environ["GITHUB_RUN_ID"]
    url = f"https://api.github.com/repos/huggingface/transformers/actions/runs/{run_id}/jobs?per_page=100"
    result = requests.get(url).json()
    jobs = {}

    try:
        jobs.update({job["name"]: job["html_url"] for job in result["jobs"]})
        pages_to_iterate_over = math.ceil((result["total_count"] - 100) / 100)

        for i in range(pages_to_iterate_over):
            result = requests.get(url + f"&page={i + 2}").json()
            jobs.update({job["name"]: job["html_url"] for job in result["jobs"]})

        return jobs
    except Exception as e:
        print("Unknown error, could not fetch links.", e)

    return {}


def retrieve_artifact(name: str):
    _artifact = {}

    if os.path.exists(name):
        files = os.listdir(name)
        for file in files:
            try:
                with open(os.path.join(name, file), encoding="utf-8") as f:
                    _artifact[file.split(".")[0]] = f.read()
            except UnicodeDecodeError as e:
                raise ValueError(f"Could not open {os.path.join(name, file)}.") from e

    return _artifact


def retrieve_available_artifacts():
    class Artifact:
        def __init__(self, name: str):
            self.name = name
            self.paths = []

        def __str__(self):
            return self.name

        def add_path(self, path: str):
            self.paths.append({"name": self.name, "path": path})

    _available_artifacts: Dict[str, Artifact] = {}

    directories = filter(os.path.isdir, os.listdir())
    for directory in directories:
        artifact_name = directory
        if artifact_name not in _available_artifacts:
            _available_artifacts[artifact_name] = Artifact(artifact_name)

            _available_artifacts[artifact_name].add_path(directory)

    return _available_artifacts


if __name__ == "__main__":
    github_actions_job_links = get_job_links()
    available_artifacts = retrieve_available_artifacts()

    docs = collections.OrderedDict(
        [
            ("*.py", "API Examples"),
            ("*.md", "MD Examples"),
        ]
    )

    # This dict will contain all the information relative to each doc test category:
    # - failed: list of failed tests
    # - failures: dict in the format 'test': 'error_message'
    doc_test_results = {
        v: {
            "failed": [],
            "failures": {},
        }
        for v in docs.values()
    }

    # Link to the GitHub Action job
    doc_test_results["job_link"] = github_actions_job_links.get("run_doctests")

    artifact_path = available_artifacts["doc_tests_gpu_test_reports"].paths[0]
    artifact = retrieve_artifact(artifact_path["name"])
    if "stats" in artifact:
        failed, success, time_spent = handle_test_results(artifact["stats"])
        doc_test_results["failures"] = failed
        doc_test_results["success"] = success
        doc_test_results["time_spent"] = time_spent[1:-1] + ", "

        all_failures = extract_first_line_failure(artifact["failures_short"])
        for line in artifact["summary_short"].split("\n"):
            if re.search("FAILED", line):
                line = line.replace("FAILED ", "")
                line = line.split()[0].replace("\n", "")

                if "::" in line:
                    file_path, test = line.split("::")
                else:
                    file_path, test = line, line

                for file_regex in docs.keys():
                    if fnmatch(file_path, file_regex):
                        category = docs[file_regex]
                        doc_test_results[category]["failed"].append(test)

                        failure = all_failures[test] if test in all_failures else "N/A"
                        doc_test_results[category]["failures"][test] = failure
                        break

    message = Message("🤗 Results of the doc tests.", doc_test_results)
    message.post()
    message.post_reply()
