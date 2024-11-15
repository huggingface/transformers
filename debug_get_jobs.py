import math
import os
import requests
import traceback


def get_jobs(workflow_run_id, token=None):
    """Extract jobs in a GitHub Actions workflow run"""

    headers = None
    if token is not None:
        headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}

    url = f"https://api.github.com/repos/huggingface/transformers/actions/runs/{workflow_run_id}/jobs?per_page=100"
    result = requests.get(url, headers=headers).json()
    jobs = []


    jobs.extend(result["jobs"])
    pages_to_iterate_over = math.ceil((result["total_count"] - 100) / 100)

    for i in range(pages_to_iterate_over):
        result = requests.get(url + f"&page={i + 2}", headers=headers).json()
        jobs.extend(result["jobs"])

    return jobs


o1 = get_jobs(workflow_run_id="11771094526")
print(o1)
print("=" * 80)
token = os.environ["ACCESS_REPO_INFO_TOKEN"]
o2 = get_jobs(workflow_run_id="11771094526", token=token)
print(o2)