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

import ast
import collections
import functools
import json
import operator
import os
import re
import sys
import time
from typing import Any, Optional, Union

import requests
from compare_test_runs import compare_job_sets
from get_ci_error_statistics import get_jobs
from get_previous_daily_ci import get_last_daily_ci_reports, get_last_daily_ci_run, get_last_daily_ci_workflow_run_id
from huggingface_hub import HfApi
from slack_sdk import WebClient


# A map associating the job names (specified by `inputs.job` in a workflow file) with the keys of
# `additional_files`.
job_to_test_map = {
    "run_models_gpu": "Models",
    "run_trainer_and_fsdp_gpu": "Trainer & FSDP",
    "run_pipelines_torch_gpu": "PyTorch pipelines",
    "run_pipelines_tf_gpu": "TensorFlow pipelines",
    "run_examples_gpu": "Examples directory",
    "run_torch_cuda_extensions_gpu": "DeepSpeed",
    "run_quantization_torch_gpu": "Quantization",
}

# The values are used as the file names where to save the corresponding CI job results.
test_to_result_name = {
    "Models": "model",
    "Trainer & FSDP": "trainer_and_fsdp",
    "PyTorch pipelines": "torch_pipeline",
    "TensorFlow pipelines": "tf_pipeline",
    "Examples directory": "example",
    "DeepSpeed": "deepspeed",
    "Quantization": "quantization",
}

NON_MODEL_TEST_MODULES = [
    "deepspeed",
    "extended",
    "fixtures",
    "generation",
    "onnx",
    "optimization",
    "pipelines",
    "sagemaker",
    "trainer",
    "utils",
    "fsdp",
    "quantization",
]


def handle_test_results(test_results):
    expressions = test_results.split(" ")

    failed = 0
    success = 0
    errors = 0
    skipped = 0

    # When the output is short enough, the output is surrounded by = signs: "== OUTPUT =="
    # When it is too long, those signs are not present.
    # It could be `'71.60s', '(0:01:11)', '====\n'` or `'in', '35.01s', '================\n'`.
    # Let always select the one with `s`.
    time_spent = expressions[-1]
    if "=" in time_spent:
        time_spent = expressions[-2]
    if "(" in time_spent:
        time_spent = expressions[-3]

    for i, expression in enumerate(expressions):
        if "failed" in expression:
            failed += int(expressions[i - 1])
        if "errors" in expression:
            errors += int(expressions[i - 1])
        if "passed" in expression:
            success += int(expressions[i - 1])
        if "skipped" in expression:
            skipped += int(expressions[i - 1])

    return failed, errors, success, skipped, time_spent


def handle_stacktraces(test_results):
    # These files should follow the following architecture:
    # === FAILURES ===
    # <path>:<line>: Error ...
    # <path>:<line>: Error ...
    # <empty line>

    total_stacktraces = test_results.split("\n")[1:-1]
    stacktraces = []
    for stacktrace in total_stacktraces:
        try:
            line = stacktrace[: stacktrace.index(" ")].split(":")[-2]
            error_message = stacktrace[stacktrace.index(" ") :]

            stacktraces.append(f"(line {line}) {error_message}")
        except Exception:
            stacktraces.append("Cannot retrieve error message.")

    return stacktraces


def dicts_to_sum(objects: Union[dict[str, dict], list[dict]]):
    if isinstance(objects, dict):
        lists = objects.values()
    else:
        lists = objects

    # Convert each dictionary to counter
    counters = map(collections.Counter, lists)
    # Sum all the counters
    return functools.reduce(operator.add, counters)


class Message:
    def __init__(
        self,
        title: str,
        ci_title: str,
        model_results: dict,
        additional_results: dict,
        selected_warnings: Optional[list] = None,
        prev_ci_artifacts=None,
        other_ci_artifacts=None,
    ):
        self.title = title
        self.ci_title = ci_title

        # Failures and success of the modeling tests
        self.n_model_success = sum(r["success"] for r in model_results.values())
        self.n_model_single_gpu_failures = sum(dicts_to_sum(r["failed"])["single"] for r in model_results.values())
        self.n_model_multi_gpu_failures = sum(dicts_to_sum(r["failed"])["multi"] for r in model_results.values())

        # Some suites do not have a distinction between single and multi GPU.
        self.n_model_unknown_failures = sum(dicts_to_sum(r["failed"])["unclassified"] for r in model_results.values())
        self.n_model_failures = (
            self.n_model_single_gpu_failures + self.n_model_multi_gpu_failures + self.n_model_unknown_failures
        )

        # Failures and success of the additional tests
        self.n_additional_success = sum(r["success"] for r in additional_results.values())

        if len(additional_results) > 0:
            # `dicts_to_sum` uses `dicts_to_sum` which requires a non empty dictionary. Let's just add an empty entry.
            all_additional_failures = dicts_to_sum([r["failed"] for r in additional_results.values()])
            self.n_additional_single_gpu_failures = all_additional_failures["single"]
            self.n_additional_multi_gpu_failures = all_additional_failures["multi"]
            self.n_additional_unknown_gpu_failures = all_additional_failures["unclassified"]
        else:
            self.n_additional_single_gpu_failures = 0
            self.n_additional_multi_gpu_failures = 0
            self.n_additional_unknown_gpu_failures = 0

        self.n_additional_failures = (
            self.n_additional_single_gpu_failures
            + self.n_additional_multi_gpu_failures
            + self.n_additional_unknown_gpu_failures
        )

        # Results
        self.n_failures = self.n_model_failures + self.n_additional_failures
        self.n_success = self.n_model_success + self.n_additional_success
        self.n_tests = self.n_failures + self.n_success

        self.model_results = model_results
        self.additional_results = additional_results

        self.thread_ts = None

        if selected_warnings is None:
            selected_warnings = []
        self.selected_warnings = selected_warnings

        self.prev_ci_artifacts = prev_ci_artifacts
        self.other_ci_artifacts = other_ci_artifacts

    @property
    def time(self) -> str:
        all_results = [*self.model_results.values(), *self.additional_results.values()]

        time_spent = []
        for r in all_results:
            if len(r["time_spent"]):
                time_spent.extend(r["time_spent"])
        total_secs = sum(time_spent)

        hours, minutes, seconds = total_secs // 3600, (total_secs % 3600) // 60, total_secs % 60
        return f"{int(hours)}h{int(minutes)}m{int(seconds)}s"

    @property
    def header(self) -> dict:
        return {"type": "header", "text": {"type": "plain_text", "text": self.title}}

    @property
    def ci_title_section(self) -> dict:
        return {"type": "section", "text": {"type": "mrkdwn", "text": self.ci_title}}

    @property
    def no_failures(self) -> dict:
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
    def failures(self) -> dict:
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
    def warnings(self) -> dict:
        # If something goes wrong, let's avoid the CI report failing to be sent.
        button_text = "Check warnings (Link not found)"
        # Use the workflow run link
        job_link = f"https://github.com/huggingface/transformers/actions/runs/{os.environ['GITHUB_RUN_ID']}"

        for job in github_actions_jobs:
            if "Extract warnings in CI artifacts" in job["name"] and job["conclusion"] == "success":
                button_text = "Check warnings"
                # Use the actual job link
                job_link = job["html_url"]
                break

        huggingface_hub_warnings = [x for x in self.selected_warnings if "huggingface_hub" in x]
        text = f"There are {len(self.selected_warnings)} warnings being selected."
        text += f"\n{len(huggingface_hub_warnings)} of them are from `huggingface_hub`."

        return {
            "type": "section",
            "text": {
                "type": "plain_text",
                "text": text,
                "emoji": True,
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": button_text, "emoji": True},
                "url": job_link,
            },
        }

    @staticmethod
    def get_device_report(report, rjust=6):
        if "single" in report and "multi" in report:
            return f"{str(report['single']).rjust(rjust)} | {str(report['multi']).rjust(rjust)} | "
        elif "single" in report:
            return f"{str(report['single']).rjust(rjust)} | {'0'.rjust(rjust)} | "
        elif "multi" in report:
            return f"{'0'.rjust(rjust)} | {str(report['multi']).rjust(rjust)} | "

    @property
    def category_failures(self) -> dict:
        if job_name != "run_models_gpu":
            category_failures_report = ""
            return {"type": "section", "text": {"type": "mrkdwn", "text": category_failures_report}}

        model_failures = [v["failed"] for v in self.model_results.values()]

        category_failures = {}

        for model_failure in model_failures:
            for key, value in model_failure.items():
                if key not in category_failures:
                    category_failures[key] = dict(value)
                else:
                    category_failures[key]["unclassified"] += value["unclassified"]
                    category_failures[key]["single"] += value["single"]
                    category_failures[key]["multi"] += value["multi"]

        individual_reports = []
        for key, value in category_failures.items():
            device_report = self.get_device_report(value)

            if sum(value.values()):
                if device_report:
                    individual_reports.append(f"{device_report}{key}")
                else:
                    individual_reports.append(key)

        header = "Single |  Multi | Category\n"
        category_failures_report = prepare_reports(
            title="The following categories had failures", header=header, reports=individual_reports
        )

        return {"type": "section", "text": {"type": "mrkdwn", "text": category_failures_report}}

    def compute_diff_for_failure_reports(self, curr_failure_report, prev_failure_report):  # noqa
        # Remove the leading and training parts that don't contain failure count information.
        model_failures = curr_failure_report.split("\n")[3:-2]
        prev_model_failures = prev_failure_report.split("\n")[3:-2]
        entries_changed = set(model_failures).difference(prev_model_failures)

        prev_map = {}
        for f in prev_model_failures:
            items = [x.strip() for x in f.split("| ")]
            prev_map[items[-1]] = [int(x) for x in items[:-1]]

        curr_map = {}
        for f in entries_changed:
            items = [x.strip() for x in f.split("| ")]
            curr_map[items[-1]] = [int(x) for x in items[:-1]]

        diff_map = {}
        for k, v in curr_map.items():
            if k not in prev_map:
                diff_map[k] = v
            else:
                diff = [x - y for x, y in zip(v, prev_map[k])]
                if max(diff) > 0:
                    diff_map[k] = diff

        entries_changed = []
        for model_name, diff_values in diff_map.items():
            diff = [str(x) for x in diff_values]
            diff = [f"+{x}" if (x != "0" and not x.startswith("-")) else x for x in diff]
            diff = [x.rjust(9) for x in diff]
            device_report = " | ".join(diff) + " | "
            report = f"{device_report}{model_name}"
            entries_changed.append(report)
        entries_changed = sorted(entries_changed, key=lambda s: s.split("| ")[-1])

        return entries_changed

    @property
    def model_failures(self) -> list[dict]:
        # Obtain per-model failures
        def per_model_sum(model_category_dict):
            return dicts_to_sum(model_category_dict["failed"].values())

        failures = {}
        non_model_failures = {
            k: per_model_sum(v) for k, v in self.model_results.items() if sum(per_model_sum(v).values())
        }

        for k, v in self.model_results.items():
            # The keys in `model_results` may contain things like `models_vit` or `quantization_autoawq`
            # Remove the prefix to make the report cleaner.
            k = k.replace("models_", "").replace("quantization_", "")
            if k in NON_MODEL_TEST_MODULES:
                continue

            if sum(per_model_sum(v).values()):
                dict_failed = dict(v["failed"])

                # Model job has a special form for reporting
                if job_name == "run_models_gpu":
                    pytorch_specific_failures = dict_failed.pop("PyTorch")
                    tensorflow_specific_failures = dict_failed.pop("TensorFlow")
                    other_failures = dicts_to_sum(dict_failed.values())

                    failures[k] = {
                        "PyTorch": pytorch_specific_failures,
                        "TensorFlow": tensorflow_specific_failures,
                        "other": other_failures,
                    }

                else:
                    test_name = job_to_test_map[job_name]
                    specific_failures = dict_failed.pop(test_name)
                    failures[k] = {
                        test_name: specific_failures,
                    }

        model_reports = []
        other_module_reports = []

        for key, value in non_model_failures.items():
            key = key.replace("models_", "").replace("quantization_", "")

            if key in NON_MODEL_TEST_MODULES:
                device_report = self.get_device_report(value)

                if sum(value.values()):
                    if device_report:
                        report = f"{device_report}{key}"
                    else:
                        report = key

                    other_module_reports.append(report)

        for key, value in failures.items():
            # Model job has a special form for reporting
            if job_name == "run_models_gpu":
                device_report_values = [
                    value["PyTorch"]["single"],
                    value["PyTorch"]["multi"],
                    value["TensorFlow"]["single"],
                    value["TensorFlow"]["multi"],
                    sum(value["other"].values()),
                ]

            else:
                test_name = job_to_test_map[job_name]
                device_report_values = [
                    value[test_name]["single"],
                    value[test_name]["multi"],
                ]

            if sum(device_report_values):
                # This is related to `model_header` below
                rjust_width = 9 if job_name == "run_models_gpu" else 6
                device_report = " | ".join([str(x).rjust(rjust_width) for x in device_report_values]) + " | "
                report = f"{device_report}{key}"

                model_reports.append(report)

        # (Possibly truncated) reports for the current workflow run - to be sent to Slack channels
        if job_name == "run_models_gpu":
            model_header = "Single PT |  Multi PT | Single TF |  Multi TF |     Other | Category\n"
        else:
            model_header = "Single |  Multi | Category\n"

        # Used when calling `prepare_reports` below to prepare the `title` argument
        label = test_to_result_name[job_to_test_map[job_name]]

        sorted_model_reports = sorted(model_reports, key=lambda s: s.split("| ")[-1])
        model_failures_report = prepare_reports(
            title=f"These following {label} modules had failures", header=model_header, reports=sorted_model_reports
        )

        module_header = "Single |  Multi | Category\n"
        sorted_module_reports = sorted(other_module_reports, key=lambda s: s.split("| ")[-1])
        module_failures_report = prepare_reports(
            title=f"The following {label} modules had failures", header=module_header, reports=sorted_module_reports
        )

        # To be sent to Slack channels
        model_failure_sections = [{"type": "section", "text": {"type": "mrkdwn", "text": model_failures_report}}]
        model_failure_sections.append({"type": "section", "text": {"type": "mrkdwn", "text": module_failures_report}})

        # Save the complete (i.e. no truncation) failure tables (of the current workflow run)
        # (to be uploaded as artifacts)

        model_failures_report = prepare_reports(
            title=f"These following {label} modules had failures",
            header=model_header,
            reports=sorted_model_reports,
            to_truncate=False,
        )
        file_path = os.path.join(os.getcwd(), f"ci_results_{job_name}/model_failures_report.txt")
        with open(file_path, "w", encoding="UTF-8") as fp:
            fp.write(model_failures_report)

        module_failures_report = prepare_reports(
            title=f"The following {label} modules had failures",
            header=module_header,
            reports=sorted_module_reports,
            to_truncate=False,
        )
        file_path = os.path.join(os.getcwd(), f"ci_results_{job_name}/module_failures_report.txt")
        with open(file_path, "w", encoding="UTF-8") as fp:
            fp.write(module_failures_report)

        if self.prev_ci_artifacts is not None:
            # if the last run produces artifact named `ci_results_{job_name}`
            if (
                f"ci_results_{job_name}" in self.prev_ci_artifacts
                and "model_failures_report.txt" in self.prev_ci_artifacts[f"ci_results_{job_name}"]
            ):
                # Compute the difference of the previous/current (model failure) table
                prev_model_failures = self.prev_ci_artifacts[f"ci_results_{job_name}"]["model_failures_report.txt"]
                entries_changed = self.compute_diff_for_failure_reports(model_failures_report, prev_model_failures)
                if len(entries_changed) > 0:
                    # Save the complete difference
                    diff_report = prepare_reports(
                        title="Changed model modules failures",
                        header=model_header,
                        reports=entries_changed,
                        to_truncate=False,
                    )
                    file_path = os.path.join(os.getcwd(), f"ci_results_{job_name}/changed_model_failures_report.txt")
                    with open(file_path, "w", encoding="UTF-8") as fp:
                        fp.write(diff_report)

                    # To be sent to Slack channels
                    diff_report = prepare_reports(
                        title="*Changed model modules failures*",
                        header=model_header,
                        reports=entries_changed,
                    )
                    model_failure_sections.append(
                        {"type": "section", "text": {"type": "mrkdwn", "text": diff_report}},
                    )

        return model_failure_sections

    @property
    def additional_failures(self) -> dict:
        failures = {k: v["failed"] for k, v in self.additional_results.items()}
        errors = {k: v["error"] for k, v in self.additional_results.items()}

        individual_reports = []
        for key, value in failures.items():
            device_report = self.get_device_report(value)

            if sum(value.values()) or errors[key]:
                report = f"{key}"
                if errors[key]:
                    report = f"[Errored out] {report}"
                if device_report:
                    report = f"{device_report}{report}"

                individual_reports.append(report)

        header = "Single |  Multi | Category\n"
        failures_report = prepare_reports(
            title="The following non-modeling tests had failures", header=header, reports=individual_reports
        )

        return {"type": "section", "text": {"type": "mrkdwn", "text": failures_report}}

    @property
    def payload(self) -> str:
        blocks = [self.header]

        if self.ci_title:
            blocks.append(self.ci_title_section)

        if self.n_model_failures > 0 or self.n_additional_failures > 0:
            blocks.append(self.failures)

        if self.n_model_failures > 0:
            block = self.category_failures
            if block["text"]["text"]:
                blocks.append(block)

            for block in self.model_failures:
                if block["text"]["text"]:
                    blocks.append(block)

        if self.n_additional_failures > 0:
            blocks.append(self.additional_failures)

        if self.n_model_failures == 0 and self.n_additional_failures == 0:
            blocks.append(self.no_failures)

        if len(self.selected_warnings) > 0:
            blocks.append(self.warnings)

        new_failure_blocks = []
        for idx, (prev_workflow_run_id, prev_ci_artifacts) in enumerate(
            [self.prev_ci_artifacts] + self.other_ci_artifacts
        ):
            if idx == 0:
                # This is the truncated version to show on slack. For now.
                new_failure_blocks = self.get_new_model_failure_blocks(
                    prev_ci_artifacts=prev_ci_artifacts, with_header=False
                )

            # To save the list of new model failures and uploaed to hub repositories
            extra_blocks = self.get_new_model_failure_blocks(prev_ci_artifacts=prev_ci_artifacts, to_truncate=False)
            if extra_blocks:
                filename = "new_failures"
                if idx > 0:
                    filename = f"{filename}_against_{prev_workflow_run_id}"

                failure_text = extra_blocks[-1]["text"]["text"]
                file_path = os.path.join(os.getcwd(), f"ci_results_{job_name}/{filename}.txt")
                with open(file_path, "w", encoding="UTF-8") as fp:
                    fp.write(failure_text)

                # upload results to Hub dataset
                file_path = os.path.join(os.getcwd(), f"ci_results_{job_name}/{filename}.txt")
                _ = api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=f"{report_repo_folder}/ci_results_{job_name}/{filename}.txt",
                    repo_id=report_repo_id,
                    repo_type="dataset",
                    token=os.environ.get("TRANSFORMERS_CI_RESULTS_UPLOAD_TOKEN", None),
                )

                # extra processing to save to json format
                new_failed_tests = {}
                nb_new_failed_tests = 0
                for line in failure_text.split():
                    if "https://github.com/huggingface/transformers/actions/runs" in line:
                        pattern = r"<(https://github.com/huggingface/transformers/actions/runs/.+?/job/.+?)\|(.+?)>"
                        items = re.findall(pattern, line)
                    elif "tests/" in line:
                        # TODO: Improve the condition here.
                        if "tests/models/" in line or (
                            "tests/quantization/" in line and job_name == "run_quantization_torch_gpu"
                        ):
                            model = line.split("/")[2]
                        else:
                            model = line.split("/")[1]
                        if model not in new_failed_tests:
                            new_failed_tests[model] = {"single-gpu": [], "multi-gpu": []}
                        for _, device in items:
                            new_failed_tests[model][f"{device}-gpu"].append(line)
                            nb_new_failed_tests += 1
                file_path = os.path.join(os.getcwd(), f"ci_results_{job_name}/{filename}.json")
                with open(file_path, "w", encoding="UTF-8") as fp:
                    json.dump(new_failed_tests, fp, ensure_ascii=False, indent=4)

                # upload results to Hub dataset
                file_path = os.path.join(os.getcwd(), f"ci_results_{job_name}/{filename}.json")
                commit_info = api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=f"{report_repo_folder}/ci_results_{job_name}/{filename}.json",
                    repo_id=report_repo_id,
                    repo_type="dataset",
                    token=os.environ.get("TRANSFORMERS_CI_RESULTS_UPLOAD_TOKEN", None),
                )
                new_failures_url = f"https://huggingface.co/datasets/{report_repo_id}/raw/{commit_info.oid}/{report_repo_folder}/ci_results_{job_name}/{filename}.json"

                if idx == 0:
                    block = {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*There are {nb_new_failed_tests} new failed tests*\n\n(compared to previous run: <https://github.com/huggingface/transformers/actions/runs/{prev_workflow_run_id}|{prev_workflow_run_id}>)",
                        },
                        "accessory": {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Check new failures"},
                            "url": new_failures_url,
                        },
                    }
                    blocks.append(block)
                else:
                    block = {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            # TODO: We should NOT assume it's always Nvidia CI, but it's the case at this moment.
                            "text": f"*There are {nb_new_failed_tests} failed tests unique to this run*\n\n(compared to{' Nvidia CI ' if is_scheduled_ci_run else ' '}run: <https://github.com/huggingface/transformers/actions/runs/{prev_workflow_run_id}|{prev_workflow_run_id}>)",
                        },
                        "accessory": {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Check failures"},
                            "url": new_failures_url,
                        },
                    }
                    blocks.append(block)

        if diff_file_url is not None:
            block = {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Test results diff*\n\n(compared to previous run: <https://github.com/huggingface/transformers/actions/runs/{prev_workflow_run_id}|{prev_workflow_run_id}>)",
                },
                "accessory": {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Check test result diff file"},
                    "url": diff_file_url,
                },
            }
            blocks.append(block)

        if len(new_failure_blocks) > 0:
            blocks.extend(new_failure_blocks)

        return json.dumps(blocks)

    @staticmethod
    def error_out(title, ci_title="", runner_not_available=False, runner_failed=False, setup_failed=False):
        blocks = []
        title_block = {"type": "header", "text": {"type": "plain_text", "text": title}}
        blocks.append(title_block)

        if ci_title:
            ci_title_block = {"type": "section", "text": {"type": "mrkdwn", "text": ci_title}}
            blocks.append(ci_title_block)

        offline_runners = []
        if runner_not_available:
            text = "💔 CI runners are not available! Tests are not run. 😭"
            result = os.environ.get("OFFLINE_RUNNERS")
            if result is not None:
                offline_runners = json.loads(result)
        elif runner_failed:
            text = "💔 CI runners have problems! Tests are not run. 😭"
        elif setup_failed:
            text = "💔 Setup job failed. Tests are not run. 😭"
        else:
            text = "💔 There was an issue running the tests. 😭"

        error_block_1 = {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": text,
            },
        }

        text = ""
        if len(offline_runners) > 0:
            text = "\n  • " + "\n  • ".join(offline_runners)
            text = f"The following runners are offline:\n{text}\n\n"
        text += "🙏 Let's fix it ASAP! 🙏"

        error_block_2 = {
            "type": "section",
            "text": {
                "type": "plain_text",
                "text": text,
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Check Action results", "emoji": True},
                "url": f"https://github.com/huggingface/transformers/actions/runs/{os.environ['GITHUB_RUN_ID']}",
            },
        }
        blocks.extend([error_block_1, error_block_2])

        payload = json.dumps(blocks)

        print("Sending the following payload")
        print(json.dumps({"blocks": blocks}))

        client.chat_postMessage(
            channel=SLACK_REPORT_CHANNEL_ID,
            text=text,
            blocks=payload,
        )

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

    def get_reply_blocks(self, job_name, job_result, failures, device, text):
        """
        failures: A list with elements of the form {"line": full test name, "trace": error trace}
        """
        # `text` must be less than 3001 characters in Slack SDK
        # keep some room for adding "[Truncated]" when necessary
        MAX_ERROR_TEXT = 3000 - len("[Truncated]")

        failure_text = ""
        for idx, error in enumerate(failures):
            new_text = failure_text + f"*{error['line']}*\n_{error['trace']}_\n\n"
            if len(new_text) > MAX_ERROR_TEXT:
                # `failure_text` here has length <= 3000
                failure_text = failure_text + "[Truncated]"
                break
            # `failure_text` here has length <= MAX_ERROR_TEXT
            failure_text = new_text

        title = job_name
        if device is not None:
            title += f" ({device}-gpu)"

        content = {"type": "section", "text": {"type": "mrkdwn", "text": text}}

        # TODO: Make sure we always have a valid job link (or at least a way not to break the report sending)
        # Currently we get the device from a job's artifact name.
        # If a device is found, the job name should contain the device type, for example, `XXX (single-gpu)`.
        # This could be done by adding `machine_type` in a job's `strategy`.
        # (If `job_result["job_link"][device]` is `None`, we get an error: `... [ERROR] must provide a string ...`)
        if job_result["job_link"] is not None and job_result["job_link"][device] is not None:
            content["accessory"] = {
                "type": "button",
                "text": {"type": "plain_text", "text": "GitHub Action job", "emoji": True},
                "url": job_result["job_link"][device],
            }

        return [
            {"type": "header", "text": {"type": "plain_text", "text": title.upper(), "emoji": True}},
            content,
            {"type": "section", "text": {"type": "mrkdwn", "text": failure_text}},
        ]

    def get_new_model_failure_blocks(self, prev_ci_artifacts, with_header=True, to_truncate=True):
        if prev_ci_artifacts is None:
            return []

        if len(self.model_results) > 0:
            target_results = self.model_results
        else:
            target_results = self.additional_results[job_to_test_map[job_name]]

        # Make the format uniform between `model_results` and `additional_results[XXX]`
        if "failures" in target_results:
            target_results = {job_name: target_results}
        sorted_dict = sorted(target_results.items(), key=lambda t: t[0])

        job = job_to_test_map[job_name]
        prev_model_results = {}
        if (
            f"ci_results_{job_name}" in prev_ci_artifacts
            and f"{test_to_result_name[job]}_results.json" in prev_ci_artifacts[f"ci_results_{job_name}"]
        ):
            prev_model_results = json.loads(
                prev_ci_artifacts[f"ci_results_{job_name}"][f"{test_to_result_name[job]}_results.json"]
            )
            # Make the format uniform between `model_results` and `additional_results[XXX]`
            if "failures" in prev_model_results:
                prev_model_results = {job_name: prev_model_results}

        all_failure_lines = {}
        for job, job_result in sorted_dict:
            if len(job_result["failures"]):
                devices = sorted(job_result["failures"].keys(), reverse=True)
                for device in devices:
                    failures = job_result["failures"][device]
                    prev_error_lines = {}
                    if job in prev_model_results and device in prev_model_results[job]["failures"]:
                        prev_error_lines = {error["line"] for error in prev_model_results[job]["failures"][device]}

                    url = None
                    if job_result["job_link"] is not None and job_result["job_link"][device] is not None:
                        url = job_result["job_link"][device]

                    for idx, error in enumerate(failures):
                        if error["line"] in prev_error_lines:
                            continue

                        new_text = f"{error['line']}\n\n"

                        if new_text not in all_failure_lines:
                            all_failure_lines[new_text] = []

                        all_failure_lines[new_text].append(f"<{url}|{device}>" if url is not None else device)

        MAX_ERROR_TEXT = 3000 - len("[Truncated]") - len("```New failures```\n\n")
        if not to_truncate:
            MAX_ERROR_TEXT = float("inf")
        failure_text = ""
        for line, devices in all_failure_lines.items():
            new_text = failure_text + f"{'|'.join(devices)} gpu\n{line}"
            if len(new_text) > MAX_ERROR_TEXT:
                # `failure_text` here has length <= 3000
                failure_text = failure_text + "[Truncated]"
                break
            # `failure_text` here has length <= MAX_ERROR_TEXT
            failure_text = new_text

        blocks = []
        if failure_text:
            if with_header:
                blocks.append(
                    {"type": "header", "text": {"type": "plain_text", "text": "New failures", "emoji": True}}
                )
            else:
                failure_text = f"{failure_text}"
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": failure_text}})

        return blocks

    def post_reply(self):
        if self.thread_ts is None:
            raise ValueError("Can only post reply if a post has been made.")

        sorted_dict = sorted(self.model_results.items(), key=lambda t: t[0])
        for job, job_result in sorted_dict:
            if len(job_result["failures"]):
                for device, failures in job_result["failures"].items():
                    text = "\n".join(
                        sorted([f"*{k}*: {v[device]}" for k, v in job_result["failed"].items() if v[device]])
                    )

                    blocks = self.get_reply_blocks(job, job_result, failures, device, text=text)

                    print("Sending the following reply")
                    print(json.dumps({"blocks": blocks}))

                    client.chat_postMessage(
                        channel=SLACK_REPORT_CHANNEL_ID,
                        text=f"Results for {job}",
                        blocks=blocks,
                        thread_ts=self.thread_ts["ts"],
                    )

                    time.sleep(1)

        for job, job_result in self.additional_results.items():
            if len(job_result["failures"]):
                for device, failures in job_result["failures"].items():
                    blocks = self.get_reply_blocks(
                        job,
                        job_result,
                        failures,
                        device,
                        text=f"Number of failures: {job_result['failed'][device]}",
                    )

                    print("Sending the following reply")
                    print(json.dumps({"blocks": blocks}))

                    client.chat_postMessage(
                        channel=SLACK_REPORT_CHANNEL_ID,
                        text=f"Results for {job}",
                        blocks=blocks,
                        thread_ts=self.thread_ts["ts"],
                    )

                    time.sleep(1)


def retrieve_artifact(artifact_path: str, gpu: Optional[str]):
    if gpu not in [None, "single", "multi"]:
        raise ValueError(f"Invalid GPU for artifact. Passed GPU: `{gpu}`.")

    _artifact = {}

    if os.path.exists(artifact_path):
        files = os.listdir(artifact_path)
        for file in files:
            try:
                with open(os.path.join(artifact_path, file)) as f:
                    _artifact[file.split(".")[0]] = f.read()
            except UnicodeDecodeError as e:
                raise ValueError(f"Could not open {os.path.join(artifact_path, file)}.") from e

    return _artifact


def retrieve_available_artifacts():
    class Artifact:
        def __init__(self, name: str, single_gpu: bool = False, multi_gpu: bool = False):
            self.name = name
            self.single_gpu = single_gpu
            self.multi_gpu = multi_gpu
            self.paths = []

        def __str__(self):
            return self.name

        def add_path(self, path: str, gpu: Optional[str] = None):
            self.paths.append({"name": self.name, "path": path, "gpu": gpu})

    _available_artifacts: dict[str, Artifact] = {}

    directories = filter(os.path.isdir, os.listdir())
    for directory in directories:
        artifact_name = directory

        name_parts = artifact_name.split("_postfix_")
        if len(name_parts) > 1:
            artifact_name = name_parts[0]

        if artifact_name.startswith("single-gpu"):
            artifact_name = artifact_name[len("single-gpu") + 1 :]

            if artifact_name in _available_artifacts:
                _available_artifacts[artifact_name].single_gpu = True
            else:
                _available_artifacts[artifact_name] = Artifact(artifact_name, single_gpu=True)

            _available_artifacts[artifact_name].add_path(directory, gpu="single")

        elif artifact_name.startswith("multi-gpu"):
            artifact_name = artifact_name[len("multi-gpu") + 1 :]

            if artifact_name in _available_artifacts:
                _available_artifacts[artifact_name].multi_gpu = True
            else:
                _available_artifacts[artifact_name] = Artifact(artifact_name, multi_gpu=True)

            _available_artifacts[artifact_name].add_path(directory, gpu="multi")
        else:
            if artifact_name not in _available_artifacts:
                _available_artifacts[artifact_name] = Artifact(artifact_name)

            _available_artifacts[artifact_name].add_path(directory)

    return _available_artifacts


def prepare_reports(title, header, reports, to_truncate=True):
    report = ""

    MAX_ERROR_TEXT = 3000 - len("[Truncated]")
    if not to_truncate:
        MAX_ERROR_TEXT = float("inf")

    if len(reports) > 0:
        # `text` must be less than 3001 characters in Slack SDK
        # keep some room for adding "[Truncated]" when necessary

        for idx in range(len(reports)):
            _report = header + "\n".join(reports[: idx + 1])
            new_report = f"{title}:\n```\n{_report}\n```\n"
            if len(new_report) > MAX_ERROR_TEXT:
                # `report` here has length <= 3000
                report = report + "[Truncated]"
                break
            report = new_report

    return report


def pop_default(l: list[Any], i: int, default: Any) -> Any:
    try:
        return l.pop(i)
    except IndexError:
        return default


if __name__ == "__main__":
    api = HfApi()
    client = WebClient(token=os.environ["CI_SLACK_BOT_TOKEN"])

    SLACK_REPORT_CHANNEL_ID = os.environ["SLACK_REPORT_CHANNEL"]

    # runner_status = os.environ.get("RUNNER_STATUS")
    # runner_env_status = os.environ.get("RUNNER_ENV_STATUS")
    setup_status = os.environ.get("SETUP_STATUS")

    # runner_not_available = True if runner_status is not None and runner_status != "success" else False
    # runner_failed = True if runner_env_status is not None and runner_env_status != "success" else False
    # Let's keep the lines regardig runners' status (we might be able to use them again in the future)
    runner_not_available = False
    runner_failed = False
    # Some jobs don't depend (`needs`) on the job `setup`: in this case, the status of the job `setup` is `skipped`.
    setup_failed = setup_status not in ["skipped", "success"]

    org = "huggingface"
    repo = "transformers"
    repository_full_name = f"{org}/{repo}"

    # This env. variable is set in workflow file (under the job `send_results`).
    ci_event = os.environ["CI_EVENT"]

    # To find the PR number in a commit title, for example, `Add AwesomeFormer model (#99999)`
    pr_number_re = re.compile(r"\(#(\d+)\)$")

    # Add Commit/PR title with a link for push CI
    # (check the title in 2 env. variables - depending on the CI is triggered via `push` or `workflow_run` event)
    ci_title_push = os.environ.get("CI_TITLE_PUSH")
    ci_title_workflow_run = os.environ.get("CI_TITLE_WORKFLOW_RUN")
    ci_title = ci_title_push if ci_title_push else ci_title_workflow_run

    ci_sha = os.environ.get("CI_SHA")

    ci_url = None
    if ci_sha:
        ci_url = f"https://github.com/{repository_full_name}/commit/{ci_sha}"

    if ci_title is not None:
        if ci_url is None:
            raise ValueError(
                "When a title is found (`ci_title`), it means a `push` event or a `workflow_run` even (triggered by "
                "another `push` event), and the commit SHA has to be provided in order to create the URL to the "
                "commit page."
            )
        ci_title = ci_title.strip().split("\n")[0].strip()

        # Retrieve the PR title and author login to complete the report
        commit_number = ci_url.split("/")[-1]
        ci_detail_url = f"https://api.github.com/repos/{repository_full_name}/commits/{commit_number}"
        ci_details = requests.get(ci_detail_url).json()
        ci_author = ci_details["author"]["login"]

        merged_by = None
        # Find the PR number (if any) and change the url to the actual PR page.
        numbers = pr_number_re.findall(ci_title)
        if len(numbers) > 0:
            pr_number = numbers[0]
            ci_detail_url = f"https://api.github.com/repos/{repository_full_name}/pulls/{pr_number}"
            ci_details = requests.get(ci_detail_url).json()

            ci_author = ci_details["user"]["login"]
            ci_url = f"https://github.com/{repository_full_name}/pull/{pr_number}"

            merged_by = ci_details["merged_by"]["login"]

        if merged_by is None:
            ci_title = f"<{ci_url}|{ci_title}>\nAuthor: {ci_author}"
        else:
            ci_title = f"<{ci_url}|{ci_title}>\nAuthor: {ci_author} | Merged by: {merged_by}"

    elif ci_sha:
        ci_title = f"<{ci_url}|commit: {ci_sha}>"

    else:
        ci_title = ""

    # `title` will be updated at the end before calling `Message()`.
    title = f"🤗 Results of {ci_event}"
    if runner_not_available or runner_failed or setup_failed:
        Message.error_out(title, ci_title, runner_not_available, runner_failed, setup_failed)
        exit(0)

    # sys.argv[0] is always `utils/notification_service.py`.
    arguments = sys.argv[1:]
    # In our usage in `.github/workflows/slack-report.yml`, we always pass an argument when calling this script.
    # The argument could be an empty string `""` if a job doesn't depend on the job `setup`.
    if arguments[0] == "":
        job_matrix = []
    else:
        job_matrix_as_str = arguments[0]
        try:
            folder_slices = ast.literal_eval(job_matrix_as_str)
            if len(folder_slices) > 0:
                if isinstance(folder_slices[0], list):
                    # Need to change from elements like `models/bert` to `models_bert` (the ones used as artifact names).
                    job_matrix = [
                        x.replace("models/", "models_").replace("quantization/", "quantization_")
                        for folders in folder_slices
                        for x in folders
                    ]
                elif isinstance(folder_slices[0], str):
                    job_matrix = [
                        x.replace("models/", "models_").replace("quantization/", "quantization_")
                        for x in folder_slices
                    ]
        except Exception:
            Message.error_out(title, ci_title)
            raise ValueError("Errored out.")

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

    available_artifacts = retrieve_available_artifacts()

    test_categories = [
        "PyTorch",
        "TensorFlow",
        "Flax",
        "Tokenizers",
        "Pipelines",
        "Trainer",
        "ONNX",
        "Auto",
        "Quantization",
        "Unclassified",
    ]

    job_name = os.getenv("CI_TEST_JOB")
    report_name_prefix = job_name

    # This dict will contain all the information relative to each model:
    # - Failures: the total, as well as the number of failures per-category defined above
    # - Success: total
    # - Time spent: as a comma-separated list of elapsed time
    # - Failures: as a line-break separated list of errors
    matrix_job_results = {
        matrix_name: {
            "failed": {m: {"unclassified": 0, "single": 0, "multi": 0} for m in test_categories},
            "errors": 0,
            "success": 0,
            "skipped": 0,
            "time_spent": [],
            "failures": {},
            "job_link": {},
        }
        for matrix_name in job_matrix
        if f"{report_name_prefix}_{matrix_name}_test_reports" in available_artifacts
    }

    unclassified_model_failures = []

    for matrix_name in matrix_job_results:
        for artifact_path_dict in available_artifacts[f"{report_name_prefix}_{matrix_name}_test_reports"].paths:
            path = artifact_path_dict["path"]
            artifact_gpu = artifact_path_dict["gpu"]

            if path not in artifact_name_to_job_map:
                # Mismatch between available artifacts and reported jobs on github. It happens.
                continue

            artifact = retrieve_artifact(path, artifact_gpu)
            if "stats" in artifact:
                # Link to the GitHub Action job
                job = artifact_name_to_job_map[path]
                matrix_job_results[matrix_name]["job_link"][artifact_gpu] = job["html_url"]
                failed, errors, success, skipped, time_spent = handle_test_results(artifact["stats"])
                matrix_job_results[matrix_name]["success"] += success
                matrix_job_results[matrix_name]["errors"] += errors
                matrix_job_results[matrix_name]["skipped"] += skipped
                matrix_job_results[matrix_name]["time_spent"].append(float(time_spent[:-1]))

                stacktraces = handle_stacktraces(artifact["failures_line"])

                # TODO: ???
                for line in artifact["summary_short"].split("\n"):
                    if line.startswith("FAILED "):
                        # Avoid the extra `FAILED` entry given by `run_test_using_subprocess` causing issue when calling
                        # `stacktraces.pop` below.
                        # See `run_test_using_subprocess` in `src/transformers/testing_utils.py`
                        if " - Failed: (subprocess)" in line:
                            continue
                        line = line[len("FAILED ") :]
                        line = line.split()[0].replace("\n", "")

                        if artifact_gpu not in matrix_job_results[matrix_name]["failures"]:
                            matrix_job_results[matrix_name]["failures"][artifact_gpu] = []

                        trace = pop_default(stacktraces, 0, "Cannot retrieve error message.")
                        matrix_job_results[matrix_name]["failures"][artifact_gpu].append(
                            {"line": line, "trace": trace}
                        )

                        # TODO: How to deal wit this

                        if re.search("tests/quantization", line):
                            matrix_job_results[matrix_name]["failed"]["Quantization"][artifact_gpu] += 1

                        elif re.search("test_modeling_tf_", line):
                            matrix_job_results[matrix_name]["failed"]["TensorFlow"][artifact_gpu] += 1

                        elif re.search("test_modeling_flax_", line):
                            matrix_job_results[matrix_name]["failed"]["Flax"][artifact_gpu] += 1

                        elif re.search("test_modeling", line):
                            matrix_job_results[matrix_name]["failed"]["PyTorch"][artifact_gpu] += 1

                        elif re.search("test_tokenization", line):
                            matrix_job_results[matrix_name]["failed"]["Tokenizers"][artifact_gpu] += 1

                        elif re.search("test_pipelines", line):
                            matrix_job_results[matrix_name]["failed"]["Pipelines"][artifact_gpu] += 1

                        elif re.search("test_trainer", line):
                            matrix_job_results[matrix_name]["failed"]["Trainer"][artifact_gpu] += 1

                        elif re.search("onnx", line):
                            matrix_job_results[matrix_name]["failed"]["ONNX"][artifact_gpu] += 1

                        elif re.search("auto", line):
                            matrix_job_results[matrix_name]["failed"]["Auto"][artifact_gpu] += 1

                        else:
                            matrix_job_results[matrix_name]["failed"]["Unclassified"][artifact_gpu] += 1
                            unclassified_model_failures.append(line)

    # Additional runs
    additional_files = {
        "PyTorch pipelines": "run_pipelines_torch_gpu_test_reports",
        "TensorFlow pipelines": "run_pipelines_tf_gpu_test_reports",
        "Examples directory": "run_examples_gpu_test_reports",
        "DeepSpeed": "run_torch_cuda_extensions_gpu_test_reports",
    }

    if ci_event in ["push", "Nightly CI"] or ci_event.startswith("Past CI"):
        del additional_files["Examples directory"]
        del additional_files["PyTorch pipelines"]
        del additional_files["TensorFlow pipelines"]
    elif ci_event.startswith("Scheduled CI (AMD)"):
        del additional_files["TensorFlow pipelines"]
        del additional_files["DeepSpeed"]
    elif ci_event.startswith("Push CI (AMD)"):
        additional_files = {}

    report_repo_id = os.getenv("REPORT_REPO_ID")

    # if it is not a scheduled run, upload the reports to a subfolder under `report_repo_folder`
    report_repo_subfolder = ""
    if os.getenv("GITHUB_EVENT_NAME") != "schedule":
        report_repo_subfolder = f"{os.getenv('GITHUB_RUN_NUMBER')}-{os.getenv('GITHUB_RUN_ID')}"
        report_repo_subfolder = f"runs/{report_repo_subfolder}"

    workflow_run = get_last_daily_ci_run(
        token=os.environ["ACCESS_REPO_INFO_TOKEN"], workflow_run_id=os.getenv("GITHUB_RUN_ID")
    )
    workflow_run_created_time = workflow_run["created_at"]
    workflow_id = workflow_run["workflow_id"]

    report_repo_folder = workflow_run_created_time.split("T")[0]

    if report_repo_subfolder:
        report_repo_folder = f"{report_repo_folder}/{report_repo_subfolder}"

    # Remove some entries in `additional_files` if they are not concerned.
    test_name = None
    if job_name in job_to_test_map:
        test_name = job_to_test_map[job_name]
    additional_files = {k: v for k, v in additional_files.items() if k == test_name}

    additional_results = {
        key: {
            "failed": {"unclassified": 0, "single": 0, "multi": 0},
            "errors": 0,
            "success": 0,
            "skipped": 0,
            "time_spent": [],
            "error": False,
            "failures": {},
            "job_link": {},
        }
        for key in additional_files
    }

    for key in additional_results:
        # If a whole suite of test fails, the artifact isn't available.
        if additional_files[key] not in available_artifacts:
            additional_results[key]["error"] = True
            continue

        for artifact_path_dict in available_artifacts[additional_files[key]].paths:
            path = artifact_path_dict["path"]
            artifact_gpu = artifact_path_dict["gpu"]

            # Link to the GitHub Action job
            job = artifact_name_to_job_map[path]
            additional_results[key]["job_link"][artifact_gpu] = job["html_url"]

            artifact = retrieve_artifact(path, artifact_gpu)
            stacktraces = handle_stacktraces(artifact["failures_line"])

            failed, errors, success, skipped, time_spent = handle_test_results(artifact["stats"])
            additional_results[key]["failed"][artifact_gpu or "unclassified"] += failed
            additional_results[key]["success"] += success
            additional_results[key]["errors"] += errors
            additional_results[key]["skipped"] += skipped
            additional_results[key]["time_spent"].append(float(time_spent[:-1]))

            if len(artifact["errors"]):
                additional_results[key]["error"] = True

            if failed:
                for line in artifact["summary_short"].split("\n"):
                    if line.startswith("FAILED "):
                        # Avoid the extra `FAILED` entry given by `run_test_using_subprocess` causing issue when calling
                        # `stacktraces.pop` below.
                        # See `run_test_using_subprocess` in `src/transformers/testing_utils.py`
                        if " - Failed: (subprocess)" in line:
                            continue
                        line = line[len("FAILED ") :]
                        line = line.split()[0].replace("\n", "")

                        if artifact_gpu not in additional_results[key]["failures"]:
                            additional_results[key]["failures"][artifact_gpu] = []

                        trace = pop_default(stacktraces, 0, "Cannot retrieve error message.")
                        additional_results[key]["failures"][artifact_gpu].append({"line": line, "trace": trace})

    # Let's only check the warning for the model testing job. Currently, the job `run_extract_warnings` is only run
    # when `inputs.job` (in the workflow file) is `run_models_gpu`. The reason is: otherwise we need to save several
    # artifacts with different names which complicates the logic for an insignificant part of the CI workflow reporting.
    selected_warnings = []
    if job_name == "run_models_gpu":
        if "warnings_in_ci" in available_artifacts:
            directory = available_artifacts["warnings_in_ci"].paths[0]["path"]
            with open(os.path.join(directory, "selected_warnings.json")) as fp:
                selected_warnings = json.load(fp)

    if not os.path.isdir(os.path.join(os.getcwd(), f"ci_results_{job_name}")):
        os.makedirs(os.path.join(os.getcwd(), f"ci_results_{job_name}"))

    nvidia_daily_ci_workflow = "huggingface/transformers/.github/workflows/self-scheduled-caller.yml"
    amd_daily_ci_workflows = (
        "huggingface/transformers/.github/workflows/self-scheduled-amd-mi300-caller.yml",
        "huggingface/transformers/.github/workflows/self-scheduled-amd-mi325-caller.yml",
    )
    is_nvidia_daily_ci_workflow = os.environ.get("GITHUB_WORKFLOW_REF").startswith(nvidia_daily_ci_workflow)
    is_amd_daily_ci_workflow = os.environ.get("GITHUB_WORKFLOW_REF").startswith(amd_daily_ci_workflows)

    is_scheduled_ci_run = os.environ.get("GITHUB_EVENT_NAME") == "schedule"
    # For AMD workflow runs: the different AMD CI callers (MI210/MI250/MI300, etc.) are triggered by `workflow_run`
    #  event of `.github/workflows/self-scheduled-amd-caller.yml`.
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_run":
        # Get the path to the file on the runner that contains the full event webhook payload.
        event_payload_path = os.environ.get("GITHUB_EVENT_PATH")
        # Load the event payload
        with open(event_payload_path) as fp:
            event_payload = json.load(fp)
            # The event that triggers the original `workflow_run`.
            if "workflow_run" in event_payload:
                is_scheduled_ci_run = event_payload["workflow_run"]["event"] == "schedule"

    test_name_and_result_pairs = []
    if len(matrix_job_results) > 0:
        test_name = job_to_test_map[job_name]
        test_name_and_result_pairs.append((test_name, matrix_job_results))

    for test_name, result in additional_results.items():
        test_name_and_result_pairs.append((test_name, result))

    for test_name, result in test_name_and_result_pairs:
        with open(f"ci_results_{job_name}/{test_to_result_name[test_name]}_results.json", "w", encoding="UTF-8") as fp:
            json.dump(result, fp, indent=4, ensure_ascii=False)

        api.upload_file(
            path_or_fileobj=f"ci_results_{job_name}/{test_to_result_name[test_name]}_results.json",
            path_in_repo=f"{report_repo_folder}/ci_results_{job_name}/{test_to_result_name[test_name]}_results.json",
            repo_id=report_repo_id,
            repo_type="dataset",
            token=os.environ.get("TRANSFORMERS_CI_RESULTS_UPLOAD_TOKEN", None),
        )

    # Let's create a file contain job --> job link
    if len(matrix_job_results) > 0:
        target_results = matrix_job_results
    else:
        default_result = {
            "failed": {"unclassified": 0, "single": 0, "multi": 0},
            "success": 0,
            "time_spent": [],
            "error": False,
            "failures": {},
            "job_link": {},
        }

        key = job_to_test_map.get(job_name)
        target_results = additional_results.get(key, default_result) if key is not None else default_result

    # Make the format uniform between `model_results` and `additional_results[XXX]`
    if "failures" in target_results:
        target_results = {job_name: target_results}

    job_links = {}
    sorted_dict = sorted(target_results.items(), key=lambda t: t[0])
    for job, job_result in sorted_dict:
        if job.startswith("models_"):
            job = job[len("models_") :]
        elif job.startswith("quantization_"):
            job = job[len("quantization_") :]
        job_links[job] = job_result["job_link"]

    with open(f"ci_results_{job_name}/job_links.json", "w", encoding="UTF-8") as fp:
        json.dump(job_links, fp, indent=4, ensure_ascii=False)

    api.upload_file(
        path_or_fileobj=f"ci_results_{job_name}/job_links.json",
        path_in_repo=f"{report_repo_folder}/ci_results_{job_name}/job_links.json",
        repo_id=report_repo_id,
        repo_type="dataset",
        token=os.environ.get("TRANSFORMERS_CI_RESULTS_UPLOAD_TOKEN", None),
    )

    prev_workflow_run_id = None
    other_workflow_run_ids = []

    if is_scheduled_ci_run:
        prev_workflow_run_id = get_last_daily_ci_workflow_run_id(
            token=os.environ["ACCESS_REPO_INFO_TOKEN"], workflow_id=workflow_id
        )
        # For a scheduled run that is not the Nvidia's scheduled daily CI, add Nvidia's scheduled daily CI run as a target to compare.
        if not is_nvidia_daily_ci_workflow:
            # The id of the workflow `.github/workflows/self-scheduled-caller.yml` (not of a workflow run of it).
            other_workflow_id = "90575235"
            # We need to get the Nvidia's scheduled daily CI run that match the current run (i.e. run with the same commit SHA)
            other_workflow_run_id = get_last_daily_ci_workflow_run_id(
                token=os.environ["ACCESS_REPO_INFO_TOKEN"], workflow_id=other_workflow_id, commit_sha=ci_sha
            )
            other_workflow_run_ids.append(other_workflow_run_id)
    else:
        prev_workflow_run_id = os.environ["PREV_WORKFLOW_RUN_ID"]
        other_workflow_run_id = os.environ["OTHER_WORKFLOW_RUN_ID"]
        other_workflow_run_ids.append(other_workflow_run_id)

    prev_ci_artifacts = (None, None)
    other_ci_artifacts = []

    output_dir = os.path.join(os.getcwd(), "previous_reports")
    os.makedirs(output_dir, exist_ok=True)

    for idx, target_workflow_run_id in enumerate([prev_workflow_run_id] + other_workflow_run_ids):
        if target_workflow_run_id is None or target_workflow_run_id == "":
            continue
        else:
            artifact_names = [f"ci_results_{job_name}"]
            ci_artifacts = get_last_daily_ci_reports(
                artifact_names=artifact_names,
                output_dir=output_dir,
                token=os.environ["ACCESS_REPO_INFO_TOKEN"],
                workflow_run_id=target_workflow_run_id,
            )
            if idx == 0:
                prev_ci_artifacts = (target_workflow_run_id, ci_artifacts)
            else:
                other_ci_artifacts.append((target_workflow_run_id, ci_artifacts))

    # Only for AMD at this moment.
    # TODO: put this into a method
    diff_file_url = None
    if is_amd_daily_ci_workflow:
        if not (prev_workflow_run_id is None or prev_workflow_run_id == ""):
            ci_artifacts = get_last_daily_ci_reports(
                artifact_names=None,
                output_dir=output_dir,
                token=os.environ["ACCESS_REPO_INFO_TOKEN"],
                workflow_run_id=prev_workflow_run_id,
            )

            current_artifacts = sorted([d for d in os.listdir() if os.path.isdir(d) and d.endswith("_test_reports")])
            prev_artifacts = sorted([d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.endswith("_test_reports")])  # fmt: skip

            current_artifacts_set = {}
            for d in current_artifacts:
                current_artifacts_set[d] = os.path.join(d, "summary_short.txt")

            prev_artifacts_set = {}
            for d in prev_artifacts:
                prev_artifacts_set[d] = os.path.join(output_dir, d, "summary_short.txt")

            report = compare_job_sets(prev_artifacts_set, current_artifacts_set)

            with open(f"ci_results_{job_name}/test_results_diff.json", "w") as fp:
                fp.write(report)

            # upload
            commit_info = api.upload_file(
                path_or_fileobj=f"ci_results_{job_name}/test_results_diff.json",
                path_in_repo=f"{report_repo_folder}/ci_results_{job_name}/test_results_diff.json",
                repo_id=report_repo_id,
                repo_type="dataset",
                token=os.environ.get("TRANSFORMERS_CI_RESULTS_UPLOAD_TOKEN", None),
            )
            diff_file_url = f"https://huggingface.co/datasets/{report_repo_id}/resolve/{commit_info.oid}/{report_repo_folder}/ci_results_{job_name}/test_results_diff.json"

    ci_name_in_report = ""
    if job_name in job_to_test_map:
        ci_name_in_report = job_to_test_map[job_name]

    title = f"🤗 Results of {ci_event}: {ci_name_in_report}"

    message = Message(
        title,
        ci_title,
        matrix_job_results,
        additional_results,
        selected_warnings=selected_warnings,
        prev_ci_artifacts=prev_ci_artifacts,
        other_ci_artifacts=other_ci_artifacts,
    )

    # send report only if there is any failure (for push CI)
    if message.n_failures or (ci_event != "push" and not ci_event.startswith("Push CI (AMD)")):
        message.post()
        message.post_reply()
