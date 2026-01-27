import os
import zipfile

import requests
from get_ci_error_statistics import download_artifact, get_artifacts_links


def get_daily_ci_runs(token, num_runs=7, workflow_id=None):
    """Get the workflow runs of the scheduled (daily) CI.

    This only selects the runs triggered by the `schedule` event on the `main` branch.
    """
    headers = None
    if token is not None:
        headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}

    # The id of a workflow (not of a workflow run).
    # From a given workflow run (where we have workflow run id), we can get the workflow id by going to
    # https://api.github.com/repos/huggingface/transformers/actions/runs/{workflow_run_id}
    # and check the `workflow_id` key.

    if not workflow_id:
        workflow_run_id = os.environ["GITHUB_RUN_ID"]
        workflow_run = requests.get(
            f"https://api.github.com/repos/huggingface/transformers/actions/runs/{workflow_run_id}", headers=headers
        ).json()
        workflow_id = workflow_run["workflow_id"]

    url = f"https://api.github.com/repos/huggingface/transformers/actions/workflows/{workflow_id}/runs"
    # On `main` branch + event being `schedule` + not returning PRs + only `num_runs` results
    url += f"?branch=main&exclude_pull_requests=true&per_page={num_runs}"

    result = requests.get(f"{url}&event=schedule", headers=headers).json()
    workflow_runs = result["workflow_runs"]
    if len(workflow_runs) == 0:
        result = requests.get(f"{url}&event=workflow_run", headers=headers).json()
        workflow_runs = result["workflow_runs"]

    return workflow_runs


def get_last_daily_ci_run(token, workflow_run_id=None, workflow_id=None, commit_sha=None):
    """Get the last completed workflow run id of the scheduled (daily) CI."""
    headers = None
    if token is not None:
        headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}

    workflow_run = None
    if workflow_run_id is not None and workflow_run_id != "":
        workflow_run = requests.get(
            f"https://api.github.com/repos/huggingface/transformers/actions/runs/{workflow_run_id}", headers=headers
        ).json()
        return workflow_run

    workflow_runs = get_daily_ci_runs(token, workflow_id=workflow_id)
    for run in workflow_runs:
        if commit_sha in [None, ""] and run["status"] == "completed":
            workflow_run = run
            break
        # if `commit_sha` is specified, return the latest completed run with `workflow_run["head_sha"]` matching the specified sha.
        elif commit_sha not in [None, ""] and run["head_sha"] == commit_sha and run["status"] == "completed":
            workflow_run = run
            break

    return workflow_run


def get_last_daily_ci_workflow_run_id(token, workflow_run_id=None, workflow_id=None, commit_sha=None):
    """Get the last completed workflow run id of the scheduled (daily) CI."""
    if workflow_run_id is not None and workflow_run_id != "":
        return workflow_run_id

    workflow_run = get_last_daily_ci_run(token, workflow_id=workflow_id, commit_sha=commit_sha)
    workflow_run_id = None
    if workflow_run is not None:
        workflow_run_id = workflow_run["id"]

    return workflow_run_id


def get_last_daily_ci_run_commit(token, workflow_run_id=None, workflow_id=None, commit_sha=None):
    """Get the commit sha of the last completed scheduled daily CI workflow run."""
    workflow_run = get_last_daily_ci_run(
        token, workflow_run_id=workflow_run_id, workflow_id=workflow_id, commit_sha=commit_sha
    )
    workflow_run_head_sha = None
    if workflow_run is not None:
        workflow_run_head_sha = workflow_run["head_sha"]

    return workflow_run_head_sha


def get_last_daily_ci_artifacts(
    output_dir,
    token,
    workflow_run_id=None,
    workflow_id=None,
    commit_sha=None,
    artifact_names=None,
):
    """Get the artifacts of last completed workflow run id of the scheduled (daily) CI."""
    workflow_run_id = get_last_daily_ci_workflow_run_id(
        token, workflow_run_id=workflow_run_id, workflow_id=workflow_id, commit_sha=commit_sha
    )
    if workflow_run_id is not None:
        artifacts_links = get_artifacts_links(workflow_run_id=workflow_run_id, token=token)

        if artifact_names is None:
            artifact_names = artifacts_links.keys()

        downloaded_artifact_names = []
        for artifact_name in artifact_names:
            if artifact_name in artifacts_links:
                artifact_url = artifacts_links[artifact_name]
                download_artifact(
                    artifact_name=artifact_name, artifact_url=artifact_url, output_dir=output_dir, token=token
                )
                downloaded_artifact_names.append(artifact_name)

        return downloaded_artifact_names


def get_last_daily_ci_reports(
    output_dir,
    token,
    workflow_run_id=None,
    workflow_id=None,
    commit_sha=None,
    artifact_names=None,
):
    """Get the artifacts' content of the last completed workflow run id of the scheduled (daily) CI."""
    downloaded_artifact_names = get_last_daily_ci_artifacts(
        output_dir,
        token,
        workflow_run_id=workflow_run_id,
        workflow_id=workflow_id,
        commit_sha=commit_sha,
        artifact_names=artifact_names,
    )

    results = {}
    for artifact_name in downloaded_artifact_names:
        artifact_zip_path = os.path.join(output_dir, f"{artifact_name}.zip")
        if os.path.isfile(artifact_zip_path):
            target_dir = os.path.join(output_dir, artifact_name)
            with zipfile.ZipFile(artifact_zip_path) as z:
                z.extractall(target_dir)

            results[artifact_name] = {}
            filename = os.listdir(target_dir)
            for filename in filename:
                file_path = os.path.join(target_dir, filename)
                if not os.path.isdir(file_path):
                    # read the file
                    with open(file_path) as fp:
                        content = fp.read()
                        results[artifact_name][filename] = content

    return results
