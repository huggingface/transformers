import os
import zipfile
from typing import Union

import requests
from get_ci_error_statistics import download_artifact, get_artifacts_links


# Default scheduled CI id
# DEFAULT_WF_ID: str = "77490895"
DEFAULT_WF_ID: str = "90575235"


def get_workflow_runs(*, workflow_id: str, token: str, num_runs: int = 7) -> dict:
    """Get the workflow runs of the specified workflow.

    This only selects the runs triggered by the `schedule` event on the `main` branch.

    Args:
        token (`str`): Github access token (must be able to read actions).
        workflow_id (`str`): Id of a workflow (not workflow run id). Can be retrieved by going to
            https://api.github.com/repos/huggingface/transformers/actions/runs/{workflow_run_id}
            and checking the `workflow_id` key.
        num_runs (`int`): Amount of CI runs to retrieve.

    Returns:
        Dictionary containing workflow runs.
    """
    headers = None
    if token is not None:
        headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}

    url = f"https://api.github.com/repos/huggingface/transformers/actions/workflows/{workflow_id}/runs"
    # On `main` branch + event being `schedule` + not returning PRs + only `num_runs` results
    params = {"branch": "main", "event": "schedule", "exclude_pull_requests": "true", "per_page": num_runs}

    result = requests.get(url, params=params, headers=headers).json()

    return result["workflow_runs"]


def get_latest_workflow_run(*, workflow_id: str, token: str) -> Union[dict, None]:
    """Get the last completed run of the specified workflow."""
    for workflow_run in get_workflow_runs(workflow_id=workflow_id, token=token):
        if workflow_run.get("status") == "completed":
            return workflow_run


def get_latest_workflow_run_id(*, workflow_id: str, token: str) -> Union[int, None]:
    """Get the id of the last completed run of the specified workflow."""
    if workflow_run := get_latest_workflow_run(workflow_id=workflow_id, token=token):
        if workflow_run_id := workflow_run.get("id"):
            try:
                return int(workflow_run_id)
            except ValueError:
                return None


def download_workflow_artifacts(*, workflow_run_id: int, artifact_names: list[str], output_dir: str, token: str):
    """Download the artifacts of the specified workflow run id."""
    artifacts_links = get_artifacts_links(worflow_run_id=workflow_run_id, token=token)
    for artifact_name in artifact_names:
        if artifact_name in artifacts_links:
            artifact_url = artifacts_links[artifact_name]
            download_artifact(
                artifact_name=artifact_name, artifact_url=artifact_url, output_dir=output_dir, token=token
            )


def get_workflow_run_reports(*, workflow_run_id: int, artifact_names: list[str], output_dir: str, token: str) -> dict[str, dict[str, str]]:
    """Get the artifacts' content of the last completed run of the specified workflow."""
    download_workflow_artifacts(
        workflow_run_id=workflow_run_id, artifact_names=artifact_names, output_dir=output_dir, token=token
    )

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


def get_latest_workflow_run_reports(
    *, workflow_id: str = DEFAULT_WF_ID, artifact_names: list[str], output_dir: str, token: str
) -> dict:
    """Get the artifacts' content of the last completed run of the specified workflow."""

    if workflow_run_id := get_latest_workflow_run_id(workflow_id=workflow_id, token=token):
        return get_workflow_run_reports(
            workflow_run_id=workflow_run_id, artifact_names=artifact_names, output_dir=output_dir, token=token
        )

    return {}
