import os
import time
import zipfile

from get_ci_error_statistics import download_artifact, get_artifacts_links, get_github_json


def get_daily_ci_runs(token, num_runs=7, workflow_id=None):
    """Get the most recent workflow runs of the scheduled (daily) CI on the main branch.

    Queries event=schedule; falls back to event=workflow_run when no results are found
    (AMD CI is triggered via workflow_run, not schedule).  Retries on stale GitHub API
    responses — see PR #48374 for background.
    """
    # Fetch current run metadata to (1) resolve workflow_id when not given, and (2) compare
    # workflow IDs to decide whether stale-cache detection applies (see stale_check below).
    current_run_id = int(os.environ.get("GITHUB_RUN_ID", 0))
    current_workflow_id = None
    if current_run_id:
        current_run = get_github_json(
            f"https://api.github.com/repos/huggingface/transformers/actions/runs/{current_run_id}", token=token
        )
        current_workflow_id = current_run["workflow_id"]
        if not workflow_id:
            workflow_id = current_workflow_id

    url = f"https://api.github.com/repos/huggingface/transformers/actions/workflows/{workflow_id}/runs"
    url += f"?branch=main&exclude_pull_requests=true&per_page={num_runs}"

    # The GitHub Actions search index (used for event=/branch= filters) can lag badly:
    # the same URL has returned total_count values of 190, 238, 311, and 413 within minutes
    # from different backend nodes (PR #48374).  We detect a stale response by checking
    # that GITHUB_RUN_ID appears in the results; if absent we retry.
    #
    # Stale-check is only active when querying the *same* workflow as the current run AND
    # the current run is schedule-triggered (so GITHUB_RUN_ID is guaranteed to be in the
    # results when the API is fresh).  The two remaining uncovered paths are left for a
    # follow-up PR once this path is confirmed stable:
    #   • AMD CI querying its own history → falls into the event=workflow_run fallback below
    #     (AMD CI is triggered via workflow_run, not schedule).
    #   • AMD CI querying the matching Nvidia run (different workflow_id) → stale_check=False.
    stale_check = (
        current_workflow_id is not None
        and int(workflow_id) == int(current_workflow_id)
        and os.environ.get("GITHUB_EVENT_NAME") == "schedule"
    )
    max_attempts = 5 if stale_check else 1

    for attempt in range(1, max_attempts + 1):
        schedule_url = f"{url}&event=schedule"
        print(f"[DEBUG get_daily_ci_runs] Querying (attempt {attempt}/{max_attempts}): {schedule_url}")
        result = get_github_json(schedule_url, token=token)
        workflow_runs = result["workflow_runs"]
        print(
            f"[DEBUG get_daily_ci_runs] event=schedule returned {len(workflow_runs)} runs (total_count={result.get('total_count')}):"
        )
        for r in workflow_runs:
            print(
                f"  id={r['id']} status={r['status']} conclusion={r.get('conclusion')} created_at={r['created_at']} event={r['event']}"
            )

        if len(workflow_runs) == 0:
            # AMD CI runs appear under event=workflow_run (triggered via the workflow_run
            # event, not schedule).  Stale-check for this path is TODO (see above).
            workflow_run_url = f"{url}&event=workflow_run"
            print(f"[DEBUG get_daily_ci_runs] Falling back to: {workflow_run_url}")
            result = get_github_json(workflow_run_url, token=token)
            workflow_runs = result["workflow_runs"]
            print(f"[DEBUG get_daily_ci_runs] event=workflow_run returned {len(workflow_runs)} runs:")
            for r in workflow_runs:
                print(
                    f"  id={r['id']} status={r['status']} conclusion={r.get('conclusion')} created_at={r['created_at']} event={r['event']}"
                )
            break

        if stale_check and current_run_id not in {r["id"] for r in workflow_runs}:
            if attempt < max_attempts:
                print(
                    f"[WARN get_daily_ci_runs] Current run {current_run_id} not found in results — "
                    f"likely a stale GitHub API cache (total_count={result.get('total_count')}). "
                    f"Retrying in 30s..."
                )
                time.sleep(30)
                continue
            else:
                print(
                    f"[WARN get_daily_ci_runs] Current run {current_run_id} still not found after "
                    f"{max_attempts} attempts — proceeding with stale results."
                )
        break

    return workflow_runs


def get_last_daily_ci_run(token, workflow_run_id=None, workflow_id=None, commit_sha=None):
    """Get the last completed workflow run id of the scheduled (daily) CI."""
    workflow_run = None
    if workflow_run_id is not None and workflow_run_id != "":
        workflow_run = get_github_json(
            f"https://api.github.com/repos/huggingface/transformers/actions/runs/{workflow_run_id}", token=token
        )
        # `get_github_json` already retries transient/rate-limit errors and raises on a failed
        # request, but guard against an unexpected payload so callers get a clear error instead of
        # a bare `KeyError` (e.g. `workflow_run["created_at"]`) further down the reporting script.
        if not isinstance(workflow_run, dict) or "created_at" not in workflow_run:
            raise RuntimeError(f"Unexpected response when fetching workflow run {workflow_run_id}: {workflow_run!r}")
        return workflow_run

    workflow_runs = get_daily_ci_runs(token, workflow_id=workflow_id)
    print(f"[DEBUG get_last_daily_ci_run] Iterating {len(workflow_runs)} runs (commit_sha={commit_sha!r}):")
    for run in workflow_runs:
        print(
            f"  checking id={run['id']} status={run['status']} conclusion={run.get('conclusion')} created_at={run['created_at']} head_sha={run['head_sha']}"
        )
        if commit_sha in [None, ""] and run["status"] == "completed":
            workflow_run = run
            break
        # if `commit_sha` is specified, return the latest completed run with `workflow_run["head_sha"]` matching the specified sha.
        elif commit_sha not in [None, ""] and run["head_sha"] == commit_sha and run["status"] == "completed":
            workflow_run = run
            break

    print(f"[DEBUG get_last_daily_ci_run] Selected run: {workflow_run['id'] if workflow_run else None}")
    return workflow_run


def get_last_daily_ci_workflow_run_id(token, workflow_run_id=None, workflow_id=None, commit_sha=None):
    """Get the last completed workflow run id of the scheduled (daily) CI."""
    print(
        f"[DEBUG get_last_daily_ci_workflow_run_id] called with workflow_run_id={workflow_run_id!r} workflow_id={workflow_id!r} commit_sha={commit_sha!r}"
    )
    if workflow_run_id is not None and workflow_run_id != "":
        print(f"[DEBUG get_last_daily_ci_workflow_run_id] returning early with workflow_run_id={workflow_run_id!r}")
        return workflow_run_id

    workflow_run = get_last_daily_ci_run(token, workflow_id=workflow_id, commit_sha=commit_sha)
    workflow_run_id = None
    if workflow_run is not None:
        workflow_run_id = workflow_run["id"]

    print(f"[DEBUG get_last_daily_ci_workflow_run_id] returning workflow_run_id={workflow_run_id!r}")
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
