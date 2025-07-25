"""An internal script to process `new_failures_with_bad_commit.json` produced by `utils/check_bad_commit.py`.

This is used by `.github/workflows/check_failed_model_tests.yml` to produce a slack report of the following form

```
<{url}|New failed tests>
{
   "GH_ydshieh": {
       "vit": 1
   }
}
```
"""

import json
import os
from collections import Counter
from copy import deepcopy

from get_previous_daily_ci import get_last_daily_ci_run
from huggingface_hub import HfApi


if __name__ == "__main__":
    api = HfApi()

    job_name = os.environ.get("JOB_NAME")

    with open("new_failures_with_bad_commit.json") as fp:
        data = json.load(fp)

    with open(f"ci_results_{job_name}/job_links.json") as fp:
        job_links = json.load(fp)

    # TODO: extend
    team_members = [
        "ydshieh",
        "zucchini-nlp",
        "ArthurZucker",
        "gante",
        "LysandreJik",
        "molbap",
        "qubvel",
        "Rocketknight1",
        "muellerzr",
        "SunMarc",
    ]

    # Counting the number of failures grouped by authors
    new_data = {}
    for model, model_result in data.items():
        for device, failed_tests in model_result.items():
            for failed_test in failed_tests:
                author = failed_test["author"]

                if author not in team_members:
                    author = failed_test["merged_by"]

                if author not in new_data:
                    new_data[author] = Counter()
                new_data[author].update([model])
    for author in new_data:
        new_data[author] = dict(new_data[author])

    # Group by author
    new_data_full = {author: deepcopy(data) for author in new_data}
    for author, _data in new_data_full.items():
        for model, model_result in _data.items():
            for device, failed_tests in model_result.items():
                # prepare job_link and add it to each entry of new failed test information.
                # need to change from `single-gpu` to `single` and same for `multi-gpu` to match `job_link`.
                key = model
                if list(job_links.keys()) == [job_name]:
                    key = job_name
                job_link = job_links[key][device.replace("-gpu", "")]

                failed_tests = [x for x in failed_tests if x["author"] == author or x["merged_by"] == author]
                for x in failed_tests:
                    x.update({"job_link": job_link})
                model_result[device] = failed_tests
            _data[model] = {k: v for k, v in model_result.items() if len(v) > 0}
        new_data_full[author] = {k: v for k, v in _data.items() if len(v) > 0}

    # Upload to Hub and get the url
    # if it is not a scheduled run, upload the reports to a subfolder under `report_repo_folder`
    report_repo_subfolder = ""
    if os.getenv("GITHUB_EVENT_NAME") != "schedule":
        report_repo_subfolder = f"{os.getenv('GITHUB_RUN_NUMBER')}-{os.getenv('GITHUB_RUN_ID')}"
        report_repo_subfolder = f"runs/{report_repo_subfolder}"

    workflow_run = get_last_daily_ci_run(
        token=os.environ["ACCESS_REPO_INFO_TOKEN"], workflow_run_id=os.getenv("GITHUB_RUN_ID")
    )
    workflow_run_created_time = workflow_run["created_at"]

    report_repo_folder = workflow_run_created_time.split("T")[0]

    if report_repo_subfolder:
        report_repo_folder = f"{report_repo_folder}/{report_repo_subfolder}"

    report_repo_id = os.getenv("REPORT_REPO_ID")

    with open("new_failures_with_bad_commit_grouped_by_authors.json", "w") as fp:
        json.dump(new_data_full, fp, ensure_ascii=False, indent=4)
    commit_info = api.upload_file(
        path_or_fileobj="new_failures_with_bad_commit_grouped_by_authors.json",
        path_in_repo=f"{report_repo_folder}/ci_results_{job_name}/new_failures_with_bad_commit_grouped_by_authors.json",
        repo_id=report_repo_id,
        repo_type="dataset",
        token=os.environ.get("TRANSFORMERS_CI_RESULTS_UPLOAD_TOKEN", None),
    )
    url = f"https://huggingface.co/datasets/{report_repo_id}/raw/{commit_info.oid}/{report_repo_folder}/ci_results_{job_name}/new_failures_with_bad_commit_grouped_by_authors.json"

    # Add `GH_` prefix as keyword mention
    output = {}
    for author, item in new_data.items():
        author = f"GH_{author}"
        output[author] = item

    report = f"<{url}|New failed tests>\\n\\n"
    report += json.dumps(output, indent=4).replace('"', '\\"').replace("\n", "\\n")
    print(report)
