import argparse
import math
import traceback

import dateutil.parser as date_parser
import requests


def extract_time_from_single_job(job):
    """Extract time info from a single job in a GitHub Actions workflow run"""

    job_info = {}

    start = job["started_at"]
    end = job["completed_at"]

    start_datetime = date_parser.parse(start)
    end_datetime = date_parser.parse(end)

    duration_in_min = round((end_datetime - start_datetime).total_seconds() / 60.0)

    job_info["started_at"] = start
    job_info["completed_at"] = end
    job_info["duration"] = duration_in_min

    return job_info


def get_job_time(workflow_run_id, token=None):
    """Extract time info for all jobs in a GitHub Actions workflow run"""

    headers = None
    if token is not None:
        headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}

    url = f"https://api.github.com/repos/huggingface/transformers/actions/runs/{workflow_run_id}/jobs?per_page=100"
    result = requests.get(url, headers=headers).json()
    job_time = {}

    try:
        job_time.update({job["name"]: extract_time_from_single_job(job) for job in result["jobs"]})
        pages_to_iterate_over = math.ceil((result["total_count"] - 100) / 100)

        for i in range(pages_to_iterate_over):
            result = requests.get(url + f"&page={i + 2}", headers=headers).json()
            job_time.update({job["name"]: extract_time_from_single_job(job) for job in result["jobs"]})

        return job_time
    except Exception:
        print(f"Unknown error, could not fetch links:\n{traceback.format_exc()}")

    return {}


if __name__ == "__main__":
    r"""
    Example:

        python get_github_job_time.py --workflow_run_id 2945609517
    """

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--workflow_run_id", type=str, required=True, help="A GitHub Actions workflow run id.")
    args = parser.parse_args()

    job_time = get_job_time(args.workflow_run_id)
    job_time = dict(sorted(job_time.items(), key=lambda item: item[1]["duration"], reverse=True))

    for k, v in job_time.items():
        print(f'{k}: {v["duration"]}')
