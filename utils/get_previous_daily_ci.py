import os
import zipfile

import requests
from get_ci_error_statistics import download_artifact, get_artifacts_links


def get_daily_ci_runs(token, num_runs=7):
    """Get the workflow runs of the scheduled (daily) CI.

    This only selects the runs triggered by the `schedule` event on the `main` branch.
    """
    headers = None
    if token is not None:
        headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}

    # The id of a workflow (not of a workflow run)
    workflow_id = "636036"

    url = f"https://api.github.com/repos/huggingface/transformers/actions/workflows/{workflow_id}/runs"
    # On `main` branch + event being `schedule` + not returning PRs + only `num_runs` results
    url += f"?branch=main&event=schedule&exclude_pull_requests=true&per_page={num_runs}"

    result = requests.get(url, headers=headers).json()

    return result["workflow_runs"]


def get_last_daily_ci_runs(token):
    """Get the last completed workflow run id of the scheduled (daily) CI."""
    workflow_runs = get_daily_ci_runs(token)
    workflow_run_id = None
    for workflow_run in workflow_runs:
        if workflow_run["status"] == "completed":
            workflow_run_id = workflow_run["id"]
            break

    return workflow_run_id


def get_last_daily_ci_artifacts(artifact_names, output_dir, token):
    """Get the artifacts of last completed workflow run id of the scheduled (daily) CI."""
    workflow_run_id = get_last_daily_ci_runs(token)
    if workflow_run_id is not None:
        artifacts_links = get_artifacts_links(worflow_run_id=workflow_run_id, token=token)
        for artifact_name in artifact_names:
            if artifact_name in artifacts_links:
                artifact_url = artifacts_links[artifact_name]
                download_artifact(
                    artifact_name=artifact_name, artifact_url=artifact_url, output_dir=output_dir, token=token
                )


def get_last_daily_ci_reports(artifact_names, output_dir, token):
    """Get the artifacts' content of the last completed workflow run id of the scheduled (daily) CI."""
    get_last_daily_ci_artifacts(artifact_names, output_dir, token)

    results = {}
    for artifact_name in artifact_names:
        artifact_zip_path = os.path.join(output_dir, f"{artifact_name}.zip")
        if os.path.isfile(artifact_zip_path):
            results[artifact_name] = {}
            with zipfile.ZipFile(artifact_zip_path) as z:
                for filename in z.namelist():
                    if not os.path.isdir(filename):
                        # read the file
                        with z.open(filename) as f:
                            results[artifact_name][filename] = f.read().decode("UTF-8")

    return results
