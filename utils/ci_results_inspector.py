import argparse
import json
from get_previous_daily_ci import get_workflow_run_reports

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


DEFAULT_ARTIFACTS = [
    "ci_results_run_models_gpu",
    "ci_results_run_examples_gpu",
    "ci_results_run_pipelines_torch_gpu",
    "ci_results_run_torch_cuda_extensions_gpu",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("-a", "--workflow-a", type=str, required=True, help="GitHub Actions workflow run id A")
    parser.add_argument("-b", "--workflow-b", type=str, required=True, help="GitHub Actions workflow run id B")
    parser.add_argument("--artifacts", type=str, nargs='+', default=DEFAULT_ARTIFACTS, help="Artifact names to process")
    parser.add_argument("-op", "--operation", type=str, choices=['|', '&'], default='|', help="Artifact names to process")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to store the downloaded artifacts and other result files.",
    )
    parser.add_argument("--token", default=None, type=str, help="A token that has actions:read permission.")
    args = parser.parse_args()

    a_reports = get_workflow_run_reports(
        workflow_run_id=args.workflow_a, artifact_names=args.artifacts, output_dir=args.output_dir, token=args.token
    )
    a_unique_failures = get_unique_failures_from_reports(reports=a_reports)

    b_reports = get_workflow_run_reports(
        workflow_run_id=args.workflow_b, artifact_names=args.artifacts, output_dir=args.output_dir, token=args.token
    )
    b_unique_failures = get_unique_failures_from_reports(reports=b_reports)


    result = {}
    for key, val in a_unique_failures.items():
        if args.operation == '|' and key not in b_unique_failures.keys():
            result[key] = val

        if args.operation == '&' and key in b_unique_failures.keys():
            result[key] = val


    for key, val in sorted(result.items(), key=lambda x: x[0]):
        print(key, val)
