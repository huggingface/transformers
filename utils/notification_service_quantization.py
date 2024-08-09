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

import ast
import datetime
import json
import os
import sys
import time
from typing import Dict

from get_ci_error_statistics import get_jobs
from huggingface_hub import HfApi
from notification_service import (
    Message,
    handle_stacktraces,
    handle_test_results,
    prepare_reports,
    retrieve_artifact,
    retrieve_available_artifacts,
)
from slack_sdk import WebClient


api = HfApi()
client = WebClient(token=os.environ["CI_SLACK_BOT_TOKEN"])


class QuantizationMessage(Message):
    def __init__(
        self,
        title: str,
        results: Dict,
    ):
        self.title = title

        # Failures and success of the modeling tests
        self.n_success = sum(r["success"] for r in results.values())
        self.single_gpu_failures = sum(r["failed"]["single"] for r in results.values())
        self.multi_gpu_failures = sum(r["failed"]["multi"] for r in results.values())
        self.n_failures = self.single_gpu_failures + self.multi_gpu_failures

        self.n_tests = self.n_failures + self.n_success
        self.results = results
        self.thread_ts = None

    @property
    def payload(self) -> str:
        blocks = [self.header]

        if self.n_failures > 0:
            blocks.append(self.failures_overwiew)
            blocks.append(self.failures_detailed)

        if self.n_failures == 0:
            blocks.append(self.no_failures)

        return json.dumps(blocks)

    @property
    def time(self) -> str:
        all_results = self.results.values()
        time_spent = []
        for r in all_results:
            if len(r["time_spent"]):
                time_spent.extend([x for x in r["time_spent"].split(", ") if len(x.strip())])
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
    def failures_overwiew(self) -> Dict:
        return {
            "type": "section",
            "text": {
                "type": "plain_text",
                "text": (
                    f"There were {self.n_failures} failures, out of {self.n_tests} tests.\n"
                    f"The suite ran in {self.time}."
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
    def failures_detailed(self) -> Dict:
        failures = {k: v["failed"] for k, v in self.results.items()}

        individual_reports = []
        for key, value in failures.items():
            device_report = self.get_device_report(value)
            if sum(value.values()):
                report = f"{device_report}{key}"
                individual_reports.append(report)

        header = "Single |  Multi | Category\n"
        failures_report = prepare_reports(
            title="The following quantization tests had failures", header=header, reports=individual_reports
        )

        return {"type": "section", "text": {"type": "mrkdwn", "text": failures_report}}

    def post(self):
        payload = self.payload
        print("Sending the following payload")
        print(json.dumps({"blocks": json.loads(payload)}))

        text = f"{self.n_failures} failures out of {self.n_tests} tests," if self.n_failures else "All tests passed."

        self.thread_ts = client.chat_postMessage(
            channel=SLACK_REPORT_CHANNEL_ID,
            blocks=payload,
            text=text,
        )

    def post_reply(self):
        if self.thread_ts is None:
            raise ValueError("Can only post reply if a post has been made.")

        for job, job_result in self.results.items():
            if len(job_result["failures"]):
                for device, failures in job_result["failures"].items():
                    blocks = self.get_reply_blocks(
                        job,
                        job_result,
                        failures,
                        device,
                        text=f'Number of failures: {job_result["failed"][device]}',
                    )

                    print("Sending the following reply")
                    print(json.dumps({"blocks": blocks}))

                    client.chat_postMessage(
                        channel="#transformers-ci-daily-quantization",
                        text=f"Results for {job}",
                        blocks=blocks,
                        thread_ts=self.thread_ts["ts"],
                    )
                    time.sleep(1)


if __name__ == "__main__":
    setup_status = os.environ.get("SETUP_STATUS")
    SLACK_REPORT_CHANNEL_ID = os.environ["SLACK_REPORT_CHANNEL"]
    setup_failed = True if setup_status is not None and setup_status != "success" else False

    # This env. variable is set in workflow file (under the job `send_results`).
    ci_event = os.environ["CI_EVENT"]

    title = f"🤗 Results of the {ci_event} - {os.getenv('CI_TEST_JOB')}."

    if setup_failed:
        Message.error_out(
            title, ci_title="", runner_not_available=False, runner_failed=False, setup_failed=setup_failed
        )
        exit(0)

    arguments = sys.argv[1:][0]
    try:
        quantization_matrix = ast.literal_eval(arguments)
        # Need to change from elements like `quantization/bnb` to `quantization_bnb` (the ones used as artifact names).
        quantization_matrix = [x.replace("quantization/", "quantization_") for x in quantization_matrix]
    except SyntaxError:
        Message.error_out(title, ci_title="")
        raise ValueError("Errored out.")

    available_artifacts = retrieve_available_artifacts()

    quantization_results = {
        quant: {
            "failed": {"single": 0, "multi": 0},
            "success": 0,
            "time_spent": "",
            "failures": {},
            "job_link": {},
        }
        for quant in quantization_matrix
        if f"run_quantization_torch_gpu_{ quant }_test_reports" in available_artifacts
    }

    github_actions_jobs = get_jobs(
        workflow_run_id=os.environ["GITHUB_RUN_ID"], token=os.environ["ACCESS_REPO_INFO_TOKEN"]
    )
    github_actions_job_links = {job["name"]: job["html_url"] for job in github_actions_jobs}

    artifact_name_to_job_map = {}
    for job in github_actions_jobs:
        for step in job["steps"]:
            if step["name"].startswith("Test suite reports artifacts: "):
                artifact_name = step["name"][len("Test suite reports artifacts: ") :]
                artifact_name_to_job_map[artifact_name] = job
                break

    for quant in quantization_results.keys():
        for artifact_path in available_artifacts[f"run_quantization_torch_gpu_{ quant }_test_reports"].paths:
            artifact = retrieve_artifact(artifact_path["path"], artifact_path["gpu"])
            if "stats" in artifact:
                # Link to the GitHub Action job
                job = artifact_name_to_job_map[artifact_path["path"]]
                quantization_results[quant]["job_link"][artifact_path["gpu"]] = job["html_url"]
                failed, success, time_spent = handle_test_results(artifact["stats"])
                quantization_results[quant]["failed"][artifact_path["gpu"]] += failed
                quantization_results[quant]["success"] += success
                quantization_results[quant]["time_spent"] += time_spent[1:-1] + ", "

                stacktraces = handle_stacktraces(artifact["failures_line"])

                for line in artifact["summary_short"].split("\n"):
                    if line.startswith("FAILED "):
                        line = line[len("FAILED ") :]
                        line = line.split()[0].replace("\n", "")

                        if artifact_path["gpu"] not in quantization_results[quant]["failures"]:
                            quantization_results[quant]["failures"][artifact_path["gpu"]] = []

                        quantization_results[quant]["failures"][artifact_path["gpu"]].append(
                            {"line": line, "trace": stacktraces.pop(0)}
                        )

    job_name = os.getenv("CI_TEST_JOB")
    if not os.path.isdir(os.path.join(os.getcwd(), f"ci_results_{job_name}")):
        os.makedirs(os.path.join(os.getcwd(), f"ci_results_{job_name}"))

    with open(f"ci_results_{job_name}/quantization_results.json", "w", encoding="UTF-8") as fp:
        json.dump(quantization_results, fp, indent=4, ensure_ascii=False)

    target_workflow = "huggingface/transformers/.github/workflows/self-scheduled-caller.yml@refs/heads/main"
    is_scheduled_ci_run = os.environ.get("CI_WORKFLOW_REF") == target_workflow

    # upload results to Hub dataset (only for the scheduled daily CI run on `main`)
    if is_scheduled_ci_run:
        api.upload_file(
            path_or_fileobj=f"ci_results_{job_name}/quantization_results.json",
            path_in_repo=f"{datetime.datetime.today().strftime('%Y-%m-%d')}/ci_results_{job_name}/quantization_results.json",
            repo_id="hf-internal-testing/transformers_daily_ci",
            repo_type="dataset",
            token=os.environ.get("TRANSFORMERS_CI_RESULTS_UPLOAD_TOKEN", None),
        )

    message = QuantizationMessage(
        title,
        results=quantization_results,
    )

    message.post()
    message.post_reply()
