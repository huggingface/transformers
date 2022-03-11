# Copyright 2020 The HuggingFace Team. All rights reserved.
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
# Old script for Slack's notification service. Still here as the entire suite has not been moved to the newer implem.

import os
import re
import sys

from slack_sdk import WebClient


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


def format_for_slack(total_results, results, scheduled: bool, title: str):
    print(total_results, results)
    header = {
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": title,
            "emoji": True,
        },
    }

    if total_results["failed"] > 0:
        total = {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Failures:*\n❌ {total_results['failed']} failures."},
                {"type": "mrkdwn", "text": f"*Passed:*\n✅ {total_results['success']} tests passed."},
            ],
        }
    else:
        total = {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": "\n🌞 All tests passed."},
            ],
        }

    blocks = [header, total]

    if total_results["failed"] > 0:
        for key, result in results.items():
            print(key, result)
            blocks.append({"type": "header", "text": {"type": "plain_text", "text": key, "emoji": True}})
            blocks.append(
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Results:*\n{result['failed']} failed, {result['success']} passed.",
                        },
                        {"type": "mrkdwn", "text": f"*Time spent:*\n{result['time_spent']}"},
                    ],
                }
            )
    elif not scheduled:
        for key, result in results.items():
            blocks.append(
                {"type": "section", "fields": [{"type": "mrkdwn", "text": f"*{key}*\n{result['time_spent']}."}]}
            )

    footer = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"<https://github.com/huggingface/transformers/actions/runs/{os.environ['GITHUB_RUN_ID']}|View on GitHub>",
        },
    }

    blocks.append(footer)

    blocks = {"blocks": blocks}

    return blocks


if __name__ == "__main__":
    arguments = sys.argv[1:]

    if "scheduled" in arguments:
        arguments.remove("scheduled")
        scheduled = True
    else:
        scheduled = False

    if scheduled:
        # The scheduled run has several artifacts for each job.
        file_paths = {
            "TF Single GPU": {
                "common": "run_all_tests_tf_gpu_test_reports/[].txt",
                "pipeline": "run_all_tests_tf_gpu_test_reports/[].txt",
            },
            "Torch Single GPU": {
                "common": "run_all_tests_torch_gpu_test_reports/[].txt",
                "pipeline": "run_all_tests_torch_gpu_test_reports/[].txt",
                "examples": "run_all_tests_torch_gpu_test_reports/[].txt",
            },
            "TF Multi GPU": {
                "common": "run_all_tests_tf_multi_gpu_test_reports/[].txt",
                "pipeline": "run_all_tests_tf_multi_gpu_test_reports/[].txt",
            },
            "Torch Multi GPU": {
                "common": "run_all_tests_torch_multi_gpu_test_reports/[].txt",
                "pipeline": "run_all_tests_torch_multi_gpu_test_reports/[].txt",
            },
            "Torch Cuda Extensions Single GPU": {"common": "run_tests_torch_cuda_extensions_gpu_test_reports/[].txt"},
            "Torch Cuda Extensions Multi GPU": {
                "common": "run_tests_torch_cuda_extensions_multi_gpu_test_reports/[].txt"
            },
        }
    else:
        file_paths = {
            "TF Single GPU": {"common": "run_all_tests_tf_gpu_test_reports/[].txt"},
            "Torch Single GPU": {"common": "run_all_tests_torch_gpu_test_reports/[].txt"},
            "TF Multi GPU": {"common": "run_all_tests_tf_multi_gpu_test_reports/[].txt"},
            "Torch Multi GPU": {"common": "run_all_tests_torch_multi_gpu_test_reports/[].txt"},
            "Torch Cuda Extensions Single GPU": {"common": "run_tests_torch_cuda_extensions_gpu_test_reports/[].txt"},
            "Torch Cuda Extensions Multi GPU": {
                "common": "run_tests_torch_cuda_extensions_multi_gpu_test_reports/[].txt"
            },
        }

    client = WebClient(token=os.environ["CI_SLACK_BOT_TOKEN"])

    if not scheduled:
        channel_id = os.environ["CI_SLACK_CHANNEL_ID"]
    elif scheduled and len(arguments):
        channel_id = os.environ["CI_SLACK_CHANNEL_ID_PAST_FUTURE"]
    else:
        channel_id = os.environ["CI_SLACK_CHANNEL_ID_DAILY"]

    if scheduled:
        title = "🤗 Results of the scheduled tests."
    else:
        title = "🤗 Self-push results"

    if len(arguments):
        title = f"{arguments} " + title

    try:
        results = {}
        for job, file_dict in file_paths.items():

            # Single return value for failed/success across steps of a same job
            results[job] = {"failed": 0, "success": 0, "time_spent": "", "failures": ""}

            for key, file_path in file_dict.items():
                try:
                    with open(file_path.replace("[]", "stats")) as f:
                        failed, success, time_spent = handle_test_results(f.read())
                        results[job]["failed"] += failed
                        results[job]["success"] += success
                        results[job]["time_spent"] += time_spent[1:-1] + ", "
                    with open(file_path.replace("[]", "summary_short")) as f:
                        for line in f:
                            if re.search("FAILED", line):
                                results[job]["failures"] += line
                except FileNotFoundError:
                    print("Artifact was not found, job was probably canceled.")

            # Remove the trailing ", "
            results[job]["time_spent"] = results[job]["time_spent"][:-2]

        test_results_keys = ["failed", "success"]
        total = {"failed": 0, "success": 0}
        for job, job_result in results.items():
            for result_key in test_results_keys:
                total[result_key] += job_result[result_key]

        if total["failed"] != 0 or scheduled:
            to_be_sent_to_slack = format_for_slack(total, results, scheduled, title)

            result = client.chat_postMessage(
                channel=channel_id,
                blocks=to_be_sent_to_slack["blocks"],
            )

        for job, job_result in results.items():
            if len(job_result["failures"]):
                client.chat_postMessage(
                    channel=channel_id, text=f"{job}\n{job_result['failures']}", thread_ts=result["ts"]
                )

    except Exception as e:
        # Voluntarily catch every exception and send it to Slack.
        raise Exception(f"Setup error: no artifacts were found. Error: {e}") from e
