import json
from pathlib import Path
from xml.etree import ElementTree as ET

from utils.process_circleci_workflow_test_reports import process_circleci_workflow


class _FakeResponse:
    def __init__(self, *, text: str | None = None, json_data: dict | None = None, status_code: int = 200):
        self.text = text or ""
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        if self._json_data is None:
            raise ValueError("No JSON payload in fake response.")
        return self._json_data


def _build_artifacts_from_junit(junit_path: Path):
    tree = ET.parse(junit_path)
    failures = []
    for testcase in tree.findall(".//testcase"):
        failure = testcase.find("failure")
        if failure is None:
            continue
        classname = testcase.attrib.get("classname", "")
        class_name = classname.split(".")[-1]
        file_path = testcase.attrib["file"]
        nodeid = f"{file_path}::{class_name}::{testcase.attrib['name']}"
        failure_msg = failure.attrib.get("message", "").strip() or (failure.text or "").strip()
        failures.append((nodeid, failure_msg))
    return failures


def test_failure_summary_generated_from_junit_fixture(tmp_path, monkeypatch):
    tests_dir = Path(__file__).resolve().parents[1]
    junit_path = tests_dir / "fixtures" / "circleci" / "junit_sample.xml"
    junit_failures = _build_artifacts_from_junit(junit_path)

    summary_lines = [f"FAILED {nodeid} - {message}" for nodeid, message in junit_failures]
    failure_lines = [f"{nodeid}: {message}" for nodeid, message in junit_failures]

    # Add a synthetic failure under tests/models to exercise the per-model aggregation.
    model_test = "tests/models/bert/test_modeling_bert.py::BertModelTest::test_forward"
    model_error = "AssertionError: logits mismatch"
    summary_lines.append(f"FAILED {model_test} - {model_error}")
    failure_lines.append(f"{model_test}: {model_error}")

    summary_short_text = "\n".join(summary_lines)
    failures_line_text = "\n".join(failure_lines)

    workflow_response = {
        "items": [
            {
                "project_slug": "gh/huggingface/transformers",
                "job_number": 42,
                "name": "tests_torch",
            }
        ]
    }
    artifacts_response = {
        "items": [
            {"path": "reports/tests_torch/summary_short.txt", "url": "https://example.com/summary", "node_index": 0},
            {"path": "reports/tests_torch/failures_line.txt", "url": "https://example.com/failures", "node_index": 0},
        ]
    }

    def fake_get(url, headers=None):
        if url.endswith("/workflow/test-workflow/job"):
            return _FakeResponse(json_data=workflow_response)
        if url.endswith("/project/gh/huggingface/transformers/42/artifacts"):
            return _FakeResponse(json_data=artifacts_response)
        if url == "https://example.com/summary":
            return _FakeResponse(text=summary_short_text)
        if url == "https://example.com/failures":
            return _FakeResponse(text=failures_line_text)
        raise AssertionError(f"Unexpected URL requested: {url}")

    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "outputs"
    process_circleci_workflow(
        "test-workflow",
        output_dir=str(output_dir),
        request_get=fake_get,
    )

    failure_summary_path = output_dir / "failure_summary.json"
    assert failure_summary_path.is_file()

    with open(failure_summary_path) as fp:
        failure_summary = json.load(fp)

    assert len(failure_summary["failures"]) == len(summary_lines)

    sample_test = junit_failures[0][0]
    assert sample_test in failure_summary["by_test"]
    assert failure_summary["by_test"][sample_test]["count"] == 1
    error_key = f"{sample_test}: {junit_failures[0][1]}"
    assert error_key in failure_summary["by_test"][sample_test]["errors"]
    assert sample_test in failure_summary["by_test"][sample_test]["variants"]

    assert "bert" in failure_summary["by_model"]
    assert failure_summary["by_model"]["bert"]["count"] == 1
    model_error_key = f"{model_test}: {model_error}"
    assert failure_summary["by_model"]["bert"]["errors"][model_error_key] == 1

    failure_summary_md = output_dir / "failure_summary.md"
    assert failure_summary_md.is_file()
    md_contents = failure_summary_md.read_text()
    assert "Failure summary" in md_contents
    assert "tests/models/bert/test_modeling_bert.py" in md_contents
