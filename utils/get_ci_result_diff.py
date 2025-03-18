import argparse
import json
from pprint import pprint
from get_previous_daily_ci import get_workflow_run_reports, get_latest_workflow_run_reports

def safe_json_loads(s):
    try:
        return json.loads(s)
    except Exception:
        return None

def get_unique_failures_from_reports(*, reports: dict[str, dict[str, str]]) -> dict[str, str]:
    unique_failures = {}
    for key, artifact_dict in reports.items():
        for artifact_name, artifact_str in artifact_dict.items():
            if artifact_name == "model_results.json":
                if artifact := safe_json_loads(artifact_str):
                    for model_name, results in artifact.items():
                        if failures := results.get("failures"):
                            for device, failed_tests in failures.items():
                                for failed_test in failed_tests:
                                    if line := failed_test.get("line"):
                                       if trace := failed_test.get("trace"):
                                           unique_failures[line] = trace
    return unique_failures

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--workflow_id", type=str, required=True, help="A GitHub Actions workflow run id.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to store the downloaded artifacts and other result files.",
    )
    parser.add_argument("--token", default=None, type=str, help="A token that has actions:read permission.")
    args = parser.parse_args()

    artifact_names = [
        "ci_results_run_models_gpu",
        #"ci_results_run_examples_gpu",
        #"ci_results_run_pipelines_torch_gpu",
        #"ci_results_run_torch_cuda_extensions_gpu",
    ]
    reports = get_workflow_run_reports(
        workflow_run_id=args.workflow_id, artifact_names=artifact_names, output_dir=args.output_dir, token=args.token
    )

    unique_failures = get_unique_failures_from_reports(reports=reports)


    other_output_dir = "/Users/ivar/dev/hf/transformers-ci-error-statistics/test2"
    workflow_run_id = 13755547860
    other_reports = get_workflow_run_reports(
        workflow_run_id=workflow_run_id, artifact_names=artifact_names, output_dir=other_output_dir, token=args.token
    )
    other_unique_failures = get_unique_failures_from_reports(reports=other_reports)


    #diff = unique_failures - other_unique_failures
    #print("unique_failures - other_unique_failures")
    #pprint(diff)
    #print()

    diff = {}
    for key, val in other_unique_failures.items():
        if key not in unique_failures.keys():
            diff[key] = val

    print("other_unique_failures - unique_failures")
    for key, val in sorted(diff.items(), key=lambda x: x[0]):
        print(key, val)
